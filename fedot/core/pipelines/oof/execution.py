from copy import deepcopy
from dataclasses import dataclass
from time import monotonic
from typing import List, Optional, Tuple

import numpy as np
from golem.core.optimisers.timer import Timer

from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.data.merge.data_merger import DataMerger
from fedot.core.data.split.data_split import _split_input_data_by_indexes
from fedot.core.operations.model import Model
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.oof.oof_rules import OOFSplit, build_oof_splits
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_node_rules import should_update_node_parameters


@dataclass(frozen=True)
class NodeFoldResult:
    train_output: OutputData
    test_output: OutputData


class PipelineOOFExecutor:
    def __init__(self,
                 pipeline: Pipeline,
                 input_data: InputData,
                 cv_folds: int,
                 n_jobs: int = 1,
                 random_seed: int = 42,
                 time_constraint=None):
        self.pipeline = pipeline
        self.input_data = input_data
        self.cv_folds = cv_folds
        self.random_seed = random_seed
        self.splits = build_oof_splits(input_data, cv_folds=cv_folds, random_seed=random_seed)
        self.pipeline.replace_n_jobs_in_nodes(n_jobs)
        self._fit_id = 0
        self._deadline = None if time_constraint is None else monotonic() + time_constraint.total_seconds()

    def predict_oof(self, output_mode: str = 'default') -> OutputData:
        self._fit_id = 0
        with Timer() as timer:
            fold_outputs = [
                (split, self._fit_node_for_fold(
                    self.pipeline.root_node,
                    train_positions=split.train_ids,
                    test_positions=split.test_ids,
                    output_mode=output_mode,
                ).test_output)
                for split in self.splits
            ]
            result = self._merge_fold_outputs(fold_outputs, np.arange(len(self.input_data.idx)))
            self.pipeline.computation_time = round(timer.minutes_from_start, 3)

        for node in self.pipeline.nodes:
            if should_update_node_parameters(node.operation.operation_type, node.operation.metadata.tags):
                node.update_params()
        return result

    def _fit_node_for_fold(
        self,
        node: PipelineNode,
        train_positions: np.ndarray,
        test_positions: np.ndarray,
        output_mode: str = 'default',
    ) -> NodeFoldResult:
        if self._deadline is not None and monotonic() >= self._deadline:
            raise TimeoutError('OOF pipeline evaluation exceeded its time constraint')
        train_input, test_input = self._build_node_inputs(node, train_positions, test_positions)
        fold_id = self._next_fit_id()
        fitted_operation, train_output = node.operation.fit(
            params=node._parameters,
            data=train_input,
            fold_id=fold_id,
            descriptive_id=node.descriptive_id,
        )
        test_output = node.operation.predict(
            fitted_operation=fitted_operation,
            params=node._parameters,
            data=test_input,
            output_mode=output_mode,
            fold_id=fold_id,
            descriptive_id=node.descriptive_id,
        )
        return NodeFoldResult(train_output=train_output, test_output=test_output)

    def _build_node_inputs(
        self,
        node: PipelineNode,
        train_positions: np.ndarray,
        test_positions: np.ndarray,
    ) -> Tuple[InputData, InputData]:
        if not node.nodes_from:
            if node.direct_set:
                raise ValueError('OOF pipeline evaluation does not support directly assigned node data yet.')
            return (
                _split_input_data_by_indexes(self.input_data, train_positions),
                _split_input_data_by_indexes(self.input_data, test_positions),
            )

        train_outputs = []
        test_outputs = []
        parents = node._nodes_from_with_fixed_order()
        for parent in parents:
            parent_fold = self._fit_node_for_fold(parent, train_positions, test_positions)
            if isinstance(parent.operation, Model):
                train_outputs.append(self._predict_node_oof_on_subset(parent, train_positions))
            else:
                train_outputs.append(parent_fold.train_output)
            test_outputs.append(parent_fold.test_output)

        train_input = self._merge_parent_outputs(parents, train_outputs)
        test_input = self._merge_parent_outputs(parents, test_outputs)
        return train_input, test_input

    def _predict_node_oof_on_subset(self, node: PipelineNode, positions: np.ndarray) -> OutputData:
        subset = _split_input_data_by_indexes(self.input_data, positions)
        splits = build_oof_splits(
            subset,
            cv_folds=self.cv_folds,
            random_seed=self.random_seed,
        )
        fold_outputs = []
        for split in splits:
            fold_result = self._fit_node_for_fold(
                node,
                train_positions=positions[split.train_ids],
                test_positions=positions[split.test_ids],
            )
            fold_outputs.append((split, fold_result.test_output))
        return self._merge_fold_outputs(fold_outputs, positions)

    @staticmethod
    def _merge_parent_outputs(parents: List[PipelineNode], outputs: List[OutputData]) -> InputData:
        merged = DataMerger.get(outputs).merge()
        merged.supplementary_data.previous_operations = [parent.operation.operation_type for parent in parents]
        return merged

    def _merge_fold_outputs(
        self,
        fold_outputs: List[Tuple[OOFSplit, OutputData]],
        result_positions: np.ndarray,
    ) -> OutputData:
        positions = np.concatenate([split.test_ids for split, _ in fold_outputs])
        order = np.argsort(positions)
        outputs = [output for _, output in fold_outputs]

        predict = self._concat_by_positions([output.predict for output in outputs], order)
        features = self._concat_by_positions(
            [output.features for output in outputs], order) if outputs[0].features is not None else None
        categorical_features = self._concat_by_positions(
            [output.categorical_features for output in outputs], order
        ) if outputs[0].categorical_features is not None else None
        result_positions = np.asarray(result_positions, dtype=int)
        first_output = outputs[0]
        return OutputData(
            idx=np.take(np.asarray(self.input_data.idx), result_positions, axis=0),
            features=features,
            predict=predict,
            target=self._take_rows(self.input_data.target, result_positions),
            task=deepcopy(first_output.task),
            data_type=first_output.data_type,
            supplementary_data=deepcopy(first_output.supplementary_data),
            categorical_features=categorical_features,
            categorical_idx=first_output.categorical_idx,
            numerical_idx=first_output.numerical_idx,
            encoded_idx=first_output.encoded_idx,
            features_names=first_output.features_names,
        )

    def _next_fit_id(self) -> int:
        fit_id = self._fit_id
        self._fit_id += 1
        return fit_id

    @staticmethod
    def _concat_by_positions(values: List[Optional[np.ndarray]], order: np.ndarray) -> Optional[np.ndarray]:
        if values[0] is None:
            return None
        concatenated = np.concatenate([np.asarray(value) for value in values], axis=0)
        return concatenated[order]

    @staticmethod
    def _take_rows(data, positions: np.ndarray):
        if data is None:
            return None
        if hasattr(data, 'iloc'):
            return data.iloc[positions]
        return np.take(data, positions, axis=0)
