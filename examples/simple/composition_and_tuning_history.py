"""Show structural pipeline evolution followed by hyperparameter tuning."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.opt_history_objects.opt_history import OptHistoryLabels
from golem.core.tuning.simultaneous import SimultaneousTuner

from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import set_random_seed


def create_data() -> InputData:
    """Create a reproducible, mildly imbalanced nonlinear classification dataset."""
    rng = np.random.default_rng(42)
    samples_count = 240
    features = rng.normal(size=(samples_count, 8))

    # Add correlated features so that models must handle redundant information.
    features[:, 6] = 0.7 * features[:, 0] + 0.3 * features[:, 2] + rng.normal(
        scale=0.25, size=samples_count)
    features[:, 7] = -0.6 * features[:, 1] + 0.4 * features[:, 3] + rng.normal(
        scale=0.3, size=samples_count)

    # Combine linear, interaction, periodic and quadratic effects.
    score = (
        1.2 * features[:, 0]
        - 0.9 * features[:, 1]
        + 0.8 * features[:, 2] * features[:, 3]
        + 0.7 * np.sin(1.5 * features[:, 4])
        - 0.45 * features[:, 5] ** 2
        + 0.35 * features[:, 6]
        + rng.normal(scale=0.55, size=samples_count)
    )
    target = (score > np.quantile(score, 0.55)).astype(int)

    # Flip a small fraction of labels to avoid a perfectly separable problem.
    noisy_indices = rng.choice(samples_count, size=int(0.06 * samples_count), replace=False)
    target[noisy_indices] = 1 - target[noisy_indices]
    task = Task(TaskTypesEnum.classification)

    return InputData(
        idx=np.arange(len(features)),
        features=features,
        target=target,
        task=task,
        data_type=DataTypesEnum.table,
    )


def print_graph(title: str, graph) -> None:
    print(f"\n{title}")
    print(graph.graph_description)
    for node in graph.nodes:
        print(f"  {node.name}: {node.parameters}")


def print_history(history) -> None:
    print("\n=== COMBINED HISTORY ===")
    for generation in history.generations:
        print(f"\nGeneration {generation.generation_num}: {generation.label}")
        for index, individual in enumerate(generation):
            parent_ids = []
            if individual.parent_operator is not None:
                parent_ids = [parent.uid for parent in individual.parent_operator.parent_individuals]
            print(f"  Individual {index}: {individual.uid}")
            print(f"    fitness: {individual.fitness.values}")
            print(f"    parents: {parent_ids}")
            print(f"    graph: {individual.graph.graph_description}")
            for node in individual.graph.nodes:
                print(f"      {node.name}: {node.parameters}")


def readable_history(history) -> dict:
    """Build a portable history view with explicit structure and hyperparameters."""
    return {
        'schema_version': 1,
        'generations': [
            {
                'generation': generation.generation_num,
                'label': getattr(generation.label, 'value', generation.label),
                'metadata': generation.metadata,
                'individuals': [
                    {
                        'uid': individual.uid,
                        'fitness': list(individual.fitness.values),
                        'parent_uids': [parent.uid for parent in individual.parents],
                        'structure': {
                            'depth': individual.graph.depth,
                            'length': individual.graph.length,
                            'nodes': [
                                {
                                    'uid': str(node.uid),
                                    'operation': node.name,
                                    'parent_uids': [str(parent.uid) for parent in node.nodes_from],
                                }
                                for node in individual.graph.nodes
                            ],
                        },
                        'hyperparameters': [
                            {
                                'node_uid': str(node.uid),
                                'operation': node.name,
                                'parameters': node.parameters,
                            }
                            for node in individual.graph.nodes
                        ],
                    }
                    for individual in generation
                ],
            }
            for generation in history.generations
        ],
    }


def save_history_artifacts(history, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / 'composition_and_tuning_history.json'
    native_history_path = output_dir / 'composition_and_tuning_history_native.json'
    history.save(native_history_path)
    with history_path.open('w', encoding='utf-8') as file:
        json.dump(readable_history(history), file, indent=2, default=str)

    convergence_path = output_dir / 'history_convergence.png'
    history.show.fitness_line(
        save_path=convergence_path,
        raw_fitness=True,
        show_generation_labels=True,
    )

    print(f"\nSaved readable history JSON: {history_path}")
    print(f"Saved native reloadable history: {native_history_path}")
    print(f"Saved phase-aware convergence plot: {convergence_path}")


def main() -> None:
    set_random_seed(42)
    data = create_data()
    task = data.task
    operations = get_operations_for_task(task=task, mode='model')
    requirements = PipelineComposerRequirements(
        primary=operations,
        secondary=operations,
        max_depth=3,
        max_arity=2,
        num_of_generations=2,
        n_jobs=1,
        show_progress=False,
    )

    composer = (
        ComposerBuilder(task)
        .with_requirements(requirements)
        .with_metrics(ClassificationMetricsEnum.accuracy)
        .with_optimizer_params(GPAlgorithmParameters(pop_size=5, max_pop_size=5))
        .build()
    )
    composed_pipeline = composer.compose_pipeline(data)
    history = composer.history
    print_graph('=== BEST PIPELINE AFTER COMPOSITION ===', composed_pipeline)

    tuner = (
        TunerBuilder(task)
        .with_tuner(SimultaneousTuner)
        .with_requirements(PipelineComposerRequirements(cv_folds=2, n_jobs=1))
        .with_metric(ClassificationMetricsEnum.accuracy)
        .with_iterations(3)
        .with_history(history)
        .build(data)
    )
    tuned_pipeline = tuner.tune(deepcopy(composed_pipeline), show_progress=False)
    print_graph('=== PIPELINE AFTER HYPERPARAMETER TUNING ===', tuned_pipeline)
    print_history(history)
    save_history_artifacts(history, Path(__file__).with_name('output'))

    labels = [generation.label for generation in history.generations]
    assert any(label == OptHistoryLabels.tuning_start for label in labels)
    assert any(label == OptHistoryLabels.tuning_results for label in labels)
    composition_structures = {
        tuple(node.name for node in individual.graph.nodes)
        for generation in history.generations
        if not str(generation.label).startswith('tuning')
        for individual in generation
    }
    assert len(composition_structures) > 1
    tuning_candidates = [
        individual
        for generation in history.generations
        if str(generation.label).startswith('tuning_iteration')
        for individual in generation
    ]
    assert tuning_candidates
    assert any(
        any(node.parameters != initial_node.parameters
            for node, initial_node in zip(candidate.graph.nodes, composed_pipeline.nodes))
        for candidate in tuning_candidates
    )
    
if __name__ == '__main__':
    main()
