from fedot.core.optimisers.populational_optimizer import PopulationalOptimizer
from fedot.core.dag.graph import Graph
from fedot.core.optimisers.archive import GenerationKeeper
from fedot.core.optimisers.gp_comp.evaluation import MultiprocessingDispatcher
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.objective import GraphFunction, ObjectiveFunction
from fedot.core.optimisers.objective.objective import Objective
from fedot.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizer, GraphOptimizerParameters
from fedot.core.optimisers.timer import OptimisationTimer
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utilities.grouped_condition import GroupedCondition

class ParticleSwarmOptimizer(PopulationalOptimizer):
    def __init__(self,
                 objective: Objective,
                 initial_graphs: Sequence[Graph],
                 requirements: PipelineComposerRequirements,
                 graph_generation_params: GraphGenerationParams,
                 graph_optimizer_params: Optional['GraphOptimizerParameters'] = None):
        super().__init__(objective, initial_graphs, requirements, graph_generation_params, graph_optimizer_params)
        self.population = None
        self.generations = GenerationKeeper(self.objective, keep_n_best=requirements.keep_n_best)
        self.timer = OptimisationTimer(timeout=self.requirements.timeout)
        self.eval_dispatcher = MultiprocessingDispatcher(adapter=graph_generation_params.adapter,
                                                         timer=self.timer,
                                                         n_jobs=requirements.n_jobs,
                                                         graph_cleanup_fn=_unfit_pipeline)

        # early_stopping_generations may be None, so use some obvious max number
        max_stagnation_length = requirements.early_stopping_generations or requirements.num_of_generations
        self.stop_optimization = \
            GroupedCondition().add_condition(
                lambda: self.timer.is_time_limit_reached(self.current_generation_num),
                'Optimisation stopped: Time limit is reached'
            ).add_condition(
                lambda: self.current_generation_num >= requirements.num_of_generations + 1,
                'Optimisation stopped: Max number of generations reached'
            ).add_condition(
                lambda: self.generations.stagnation_duration >= max_stagnation_length,
                'Optimisation finished: Early stopping criteria was satisfied'
            )

    # Наша оптимизируемая функция это objective
    def _itit_population(self):
        # создаем популяцию
        population = [Individual(graph) for graph in self.initial_graphs]
        # присваиваем каждой частице координату
        particles = [[random.uniform(bound_low, bound_up) for j in range(dimension)] for i in range(len(population))]
        # присваиваем лучшему известному положению частицы его начальное значение
        particle_best_position = particles
        # считаем функцию для каждой частицы
        particle_best_fitness = [evaluator(p[0], p[1]) for p in particles]
        # индекс лучшей частицы
        global_best_index = np.argmin(particle_best_fitness)
        # лучшее глобальное положение частицы
        global_best_position = particle_best_position[global_best_index]
        # присваиваем скорость для каждой частицы
        velocity = [[0.0 for j in range(dimension)] for i in range(len(population)]

    def optimize(self, objective: ObjectiveFunction, fitness_criterion): # fitness_criterion надо както задать
        # условие останова из ParticleOptimizer
        evaluator = self.eval_dispatcher.dispatch(objective)

        with self.timer, self._progressbar:
            self._init_population(evaluator=evaluator)

            while not self.stop_optimization():
                for t in range(generation):
                    # Stop if the average fitness value reached a predefined success criterion
                    if np.average(particle_best_fitness) <= fitness_criterion:
                        break
                    else:
                        for n in range(population):
                            # обновить скорость каждой частицы
                            velocity[n] = update_velocity(particles[n], velocity[n], particle_best_fitness[n], global_best_position)
                            # обновить позицию каждой частицы
                            particles[n] = update_position(particles[n], velocity[n])
                    # считаем функцию
                    particle_best_fitness = [evaluator(p[0], p[1]) for p in particles]
                    # индекс лучшей частицы
                    global_best_index = np.argmin(particle_best_fitness)
                    # обновить позицию лучшей частицы
                    global_best_position = particle_best_position[global_best_index]

    def _update_velocity(self, particle, velocity, pbest, gbest, w_min=0.5, max=1.0, c=0.1):
        # инициализируем новый массив скоростей
        num_particle = len(particle)
        new_velocity = np.array([0.0 for i in range(num_particle)])
        # случайно генерируем r1, r2 и w (inertia weight) из нормального распределения
        r1 = random.uniform(0, max)
        r2 = random.uniform(0, max)
        w = random.uniform(w_min, max)
        c1 = c
        c2 = c
        # считаем новую скорость
        for i in range(num_particle):
            new_velocity[i] = w * velocity[i] + c1 * r1 * (pbest[i] - particle[i]) + c2 * r2 * (gbest[i] - particle[i])
        return new_velocity

    def _update_position(self, particle, velocity):
        # обновляем положение частицы переносом на вектор скорости
        new_particle = particle + velocity
        return new_particle
