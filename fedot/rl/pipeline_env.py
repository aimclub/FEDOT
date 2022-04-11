import os
from os import makedirs
from os.path import join, exists

import gym
import numpy as np
from gym import spaces
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.validation import validate
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import TaskTypesEnum, Task
from fedot.core.utils import default_fedot_data_dir, fedot_project_root


class PipelineEnv(gym.Env):
    metadata = {'render.modes': ['text', 'graph']}
    reward_range = (-float(100), float(100))
    spec = None

    def __init__(self, path_to_data=None, path_to_valid=None, pipeline_depth=4, graph_render=True):
        self.full_train_data = InputData.from_csv(path_to_data)
        self.train_data, self.test_data = train_test_data_setup(InputData.from_csv(path_to_data), split_ratio=0.7)
        self.testing_data = self.test_data.target

        if path_to_valid:
            self.valid_data = InputData.from_csv(path_to_valid)
            self.validing_data = self.valid_data.target

        self.task_type = TaskTypesEnum.classification
        self.pipeline_depth = pipeline_depth

        self.actions_list = OperationTypesRepository('all') \
            .suitable_operation(task_type=self.task_type, tags=['reinforce'])

        self.model_ops = OperationTypesRepository('model') \
            .suitable_operation(task_type=self.task_type, tags=['reinforce'])[0]

        self.data_ops = OperationTypesRepository('data_operation') \
            .suitable_operation(task_type=self.task_type, tags=['reinforce'])[0]

        self.actions_list_size = len(self.actions_list[0])
        self.actions_size = self.actions_list_size + 2  # actions list + pop + eop

        self.pop = self.actions_list_size  # Placeholder of pipeline
        self.eop = self.actions_list_size + 1  # End of pipeline

        self.nodes = []
        self.pipeline = None
        self.pipeline_idx = 0
        self.time_step = 0
        self.cur_pos = 1
        self.metric_value = 0
        self.observation = np.full(self.pipeline_depth + 1, self.pop)
        self.observation[0] = self.cur_pos

        self.action_space = spaces.Discrete(self.actions_size)
        self.observation_space = self.transform_to_one_hot(self.observation)

        if graph_render:
            self.graph_render_path = join(default_fedot_data_dir(), 'rl', 'pipelines')
            if not exists(self.graph_render_path):
                makedirs(self.graph_render_path)

        self.reset()

    def step(self, action: int, mode='train'):
        temp_reward = 0
        assert self.action_space.contains(action)

        # Штраф для агента за выбор placeholder
        if action == self.pop:
            self._update_observation(action)
            reward, done, info = self._env_response()

            return self.transform_to_one_hot(self.observation), reward, done, info

        # Штраф для агента за выбор окончания построения пайплайна в самом начале
        if action == self.eop and self.nodes == []:
            self._update_observation(action)
            reward, done, info = self._env_response()

            return self.transform_to_one_hot(self.observation), reward, done, info

        # Поощерение, если агент выбрал модель
        if action != self.eop and self.actions_list[0][action] in self.model_ops:
            temp_reward += 4

        # Поощерение, если агент выбрал операцию
        if action != self.eop and self.actions_list[0][action] in self.data_ops:
            temp_reward += 1

            # Поощерение, если ее поставил в самое начало
            if self.cur_pos in [1, 2]:
                temp_reward += 1

        self._construct_pipeline(action)

        self._update_observation(action)

        # Если агент решил закончить построение пайплайна или достиг лимита,
        # то обороачиваем и проверяем
        if action == self.eop or self.cur_pos == len(self.observation):
            self.pipeline = Pipeline(self.nodes[-1])
            pipeline = self.pipeline
            try:
                # Проверка пайплайна на ошибки построения
                validate(pipeline, task=Task(self.task_type))

                # Если прошел, то обучаем
                if mode == 'train':
                    self.pipeline.fit(self.train_data)

                    # Проверка полученого пайплайна на тестовых данных
                    results = self.pipeline.predict(self.test_data)

                    # Подсчитываем метрику
                    self.metric_value = roc_auc(y_true=self.testing_data,
                                                y_score=results.predict,
                                                multi_class='ovo',
                                                average='macro')

                elif mode == 'test':
                    self.pipeline.fit(self.full_train_data)

                    results = self.pipeline.predict(self.valid_data)

                    self.metric_value = roc_auc(y_true=self.validing_data,
                                                y_score=results.predict,
                                                multi_class='ovo',
                                                average='macro')

                    self.render(mode='graph')

                reward = 100 * self.metric_value

                _, done, info = self._env_response(length=len(self.nodes))
                return self.transform_to_one_hot(self.observation), reward, done, info
            except ValueError:
                reward, done, info = self._env_response(reward=-85)
                return self.transform_to_one_hot(self.observation), reward, done, info
        else:
            reward, done, info = self._env_response(reward=temp_reward, done=False)
            return self.transform_to_one_hot(self.observation), reward, done, info

    def reset(self):
        self.pipeline = None
        self.nodes = []
        self.time_step = 0
        self.cur_pos = 1
        self.metric_value = 0
        self.observation = np.full(self.pipeline_depth + 1, self.pop)
        self.observation[0] = self.cur_pos

        return self.transform_to_one_hot(self.observation)

    def _env_response(self, reward=-100., done=True, length=-1):
        info = {
            'time_step': self.time_step,
            'metric_value': self.metric_value,
            'length': length
        }

        return reward, done, info  # reward, done, info

    def _update_observation(self, action):
        self.observation[self.cur_pos] = action
        self.cur_pos += 1
        self.observation[0] = self.cur_pos

    def _construct_pipeline(self, action):
        if action != self.eop:
            if self.time_step == 0:
                self.nodes.append(PrimaryNode(self.actions_list[0][action]))
            else:
                self.nodes.append(SecondaryNode(self.actions_list[0][action], nodes_from=[self.nodes[-1]]))
            self.time_step += 1

    def _is_data_operation(self, placed_action):
        if placed_action != self.eop and placed_action != self.pop:
            if any(self.actions_list[0][placed_action] in op for op in self.data_ops):
                return True
        else:
            return False

    def transform_to_one_hot(self, observation):
        encoded_position = np.eye(np.max(self.pipeline_depth + 1) + 1)[observation[0]].flatten()
        encoded_observation = np.eye(np.max(self.action_space.n))[observation[1:]].flatten()

        return np.concatenate((encoded_position, encoded_observation), axis=None)

    def render(self, mode='text'):
        if mode == 'text':
            print('Pipeline', self.observation)
        elif mode == 'graph':
            if self.pipeline:
                result_path = join(self.graph_render_path, f'pl_{self.pipeline_idx}')
                self.pipeline.show(path=result_path)
                self.pipeline_idx += 1


if __name__ == '__main__':
    file_path_train = 'cases/data/scoring/scoring_train.csv'
    full_path_train = os.path.join(str(fedot_project_root()), file_path_train)

    # file_path_test = 'cases/data/scoring/scoring_test.csv'
    # full_path_test = os.path.join(str(fedot_project_root()), file_path_test)

    env = PipelineEnv(path_to_data=full_path_train)

    for episode in range(5):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            print('episode:', episode, 'state', state)
            action = int(input())
            state, reward, done, info = env.step(action)
            print('reward: %6.2f' % reward)
            total_reward += reward

            if done:
                print('episode:', episode, 'state', state)
                print('Done')
