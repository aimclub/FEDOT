import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.validation import validate
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import TaskTypesEnum, Task


class PipelineEnv(gym.Env):
    metadata = {'render.modes': ['human', 'graph']}

    def __init__(self, path, pipeline_depth=4):
        self.empty_code = 1
        self.train_data = InputData.from_csv(path[0])
        self.test_data = InputData.from_csv(path[1])
        # data = get_iris_data()
        # self.train_data, self.test_data = train_test_data_setup(data, shuffle_flag=True)
        self.testing_data = self.test_data.target
        self.pipeline_depth = pipeline_depth
        self.task_type = TaskTypesEnum.classification

        self.actions_list = OperationTypesRepository('all') \
            .suitable_operation(task_type=self.task_type, tags=['reinforce'])
        self.actions_size = len(self.actions_list[0]) + 1

        self.action_space = spaces.Discrete(self.actions_size, start=2)
        self.observation_space = spaces.MultiDiscrete(
            np.full((self.pipeline_depth, self.pipeline_depth), self.empty_code))

        self.pipeline = None
        self.nodes = []
        self.time_step = 0
        self.current_position = 0
        self.metric_value = 0
        self.observation = np.full(self.pipeline_depth, 0)

        self.seed_value = self.seed()
        self.reset()

    def render(self, mode='human'):
        if mode == 'human':
            if self.pipeline:
                self.pipeline.show()
            print('Pipeline', self.observation)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: int):
        done = False
        reward = 0

        assert self.action_space.contains(action)

        self.observation[self.current_position] = action

        # Проверка для создания первого узла (PrimaryNode)
        if self.time_step == 0:
            # Если действие
            if action != self.actions_size + 1:
                self.nodes.append(PrimaryNode(self.actions_list[0][action - 2]))
                self.current_position += 1
                self.time_step += 1
        else:
            # Для каждого последущего узла добавляем связь (SecondaryNode)
            if action != self.actions_size + 1:
                self.nodes.append(SecondaryNode(self.actions_list[0][action - 2], nodes_from=[self.nodes[-1]]))
                self.current_position += 1
                self.time_step += 1

        if action == self.actions_size + 1 and not self.nodes:
            done = True
            reward -= 150
            return self.observation, reward, done, {'time_step': self.time_step, 'metric_value': self.metric_value}

        # Если действие является точкой в пайплайне, то формируем его и проверяем
        if self.nodes and action == self.actions_size + 1 or self.current_position == len(self.observation):
            self.pipeline = Pipeline(self.nodes[-1])
            pipeline = self.pipeline
            # Проверка пайплайна
            try:
                validate(pipeline, task=Task(self.task_type))
                # pipeline.show()
                self.pipeline.fit(self.train_data)
                results = self.pipeline.predict(self.test_data)
                self.metric_value = roc_auc(y_true=self.testing_data, y_score=results.predict, multi_class='ovo',
                                            average='macro')

                reward = 100 * self.metric_value
                done = True
                return self.observation, reward, done, {'time_step': self.time_step, 'metric_value': self.metric_value}

            except ValueError:
                done = True
                reward -= 100
                return self.observation, reward, done, {'time_step': self.time_step, 'metric_value': self.metric_value}
        else:
            return self.observation, reward, done, {'time_step': self.time_step, 'metric_value': self.metric_value}

    def reset(self):
        self.pipeline = None
        self.nodes = []
        self.time_step = 0
        self.current_position = 0
        self.metric_value = 0
        self.observation = np.full(self.pipeline_depth, 1)

        return self.observation
