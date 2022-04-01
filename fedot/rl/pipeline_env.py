import os
from copy import deepcopy
from os import makedirs
from os.path import join, exists

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
from fedot.core.utils import default_fedot_data_dir, fedot_project_root


class PipelineEnv(gym.Env):
    metadata = {'render.modes': ['text', 'graph']}

    def __init__(self, path, pipeline_depth=4, graph_render=True):
        self.train_data = InputData.from_csv(path[0])
        self.test_data = InputData.from_csv(path[1])
        self.testing_data = self.test_data.target

        self.task_type = TaskTypesEnum.classification
        self.pipeline_depth = pipeline_depth + 1

        self.actions_list = OperationTypesRepository('all') \
            .suitable_operation(task_type=self.task_type, tags=['reinforce'])
        self.data_ops = OperationTypesRepository('data_operation') \
            .suitable_operation(task_type=self.task_type, tags=['reinforce'])[0]
        self.actions_list_size = len(self.actions_list[0])
        self.actions_size = self.actions_list_size + 2  # actions list + pop + eop
        self.pop = self.actions_list_size  # Placeholder of pipeline
        self.eop = self.actions_list_size + 1  # End of pipeline

        self.action_space = spaces.Discrete(self.actions_size)
        self.observation_space = spaces.MultiDiscrete(
            np.full((self.pipeline_depth, self.pipeline_depth), self.pop))

        self.nodes = []
        self.pipeline = None
        self.pipeline_idx = 0
        self.time_step = 0
        self.cur_pos = 1
        self.metric_value = 0
        self.observation = np.full(self.pipeline_depth, self.pop)
        self.observation[0] = self.cur_pos
        self.last_observation = np.full(self.pipeline_depth, self.pop)
        self.repeated_pl = 0

        if graph_render:
            self.graph_render_path = join(default_fedot_data_dir(), 'rl', 'pipelines')
            if not exists(self.graph_render_path):
                makedirs(self.graph_render_path)

        self.reset()

    def render(self, mode='text'):
        if mode == 'text':
            print('Pipeline', self.observation)
        elif mode == 'graph':
            if self.pipeline:
                result_path = join(self.graph_render_path, f'pipeline_{self.pipeline_idx}')
                self.pipeline.show(path=result_path)
                self.pipeline_idx += 1

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: int):
        done = False
        reward = 0

        assert self.action_space.contains(action)

        # Если агент выбирал плейхолдеры, то штрафуем
        if action == self.pop:
            done = True
            reward -= 100
            self.observation[self.cur_pos] = action
            self.cur_pos += 1
            self.observation[0] = self.cur_pos

            info = {
                'time_step': self.time_step,
                'metric_value': self.metric_value,
                'length': -1
            }

            return self.observation, reward, done, info

        # Если агент первым действием выбрал заокнчить построение пайплайна, то штрафуем
        if action == self.eop and self.nodes == []:
            done = True
            reward -= 100
            self.observation[self.cur_pos] = action
            self.cur_pos += 1
            self.observation[0] = self.cur_pos

            info = {'time_step': self.time_step,
                    'metric_value': self.metric_value,
                    'length': -1
                    }

            return self.observation, reward, done, info

        self.observation[self.cur_pos] = action

        # Проверка для создания первого узла (PrimaryNode)
        if self.time_step == 0:
            # Если действие
            if action < self.pop:
                self.nodes.append(PrimaryNode(self.actions_list[0][action]))
                self.cur_pos += 1
                self.time_step += 1
        else:
            # Для каждого последущего узла добавляем связь (SecondaryNode)
            if action < self.pop:
                self.nodes.append(SecondaryNode(self.actions_list[0][action], nodes_from=[self.nodes[-1]]))
                self.cur_pos += 1
                self.time_step += 1

        # Если агент решил закончить построение пайплайна или достиг лимита,
        # то обороачиваем и проверяем
        if action == self.eop or self.cur_pos == len(self.observation):
            self.pipeline = Pipeline(self.nodes[-1])
            pipeline = self.pipeline
            try:
                # Проверка пайплайна на ошибки построения
                validate(pipeline, task=Task(self.task_type))

                # Если прошел, то обучаем
                self.pipeline.fit(self.train_data)

                # Проверка полученого пайплайна на тестовых данных
                results = self.pipeline.predict(self.test_data)

                # Подсчитываем метрику
                self.metric_value = roc_auc(y_true=self.testing_data, y_score=results.predict, multi_class='ovo',
                                            average='macro')

                self.render(mode='graph')
                done = True
                reward = 100 * self.metric_value

                # Если первые узлы с предобработкой данных, то агента наказывают
                for i in [1, 2]:
                    if self._is_data_operation(self.observation[i]):
                        reward -= 5

                if np.all(self.observation == self.last_observation):
                    self.repeated_pl += 1

                    if self.repeated_pl > 10:
                        reward = -50

                    if self.repeated_pl > 20:
                        reward = -100

                else:
                    self.repeated_pl = 0

                info = {'time_step': self.time_step,
                        'metric_value': self.metric_value,
                        'length': len(self.nodes),
                        }

                self.last_observation = deepcopy(self.observation)

                self.observation[0] = self.cur_pos

                return self.observation, reward, done, info
            except ValueError:
                done = True
                reward -= 85

                # Если первые узлы с предобработкой данных, то агента поощеряют
                for i in [0, 1]:
                    if self._is_data_operation(self.observation[i]):
                        reward += 15

                if np.all(self.observation == self.last_observation):
                    self.repeated_pl += 1

                    if self.repeated_pl > 10:
                        reward -= 50
                else:
                    self.repeated_pl = 0

                info = {'time_step': self.time_step,
                        'metric_value': self.metric_value,
                        'length': -1
                        }

                self.last_observation = deepcopy(self.observation)

                self.observation[0] = self.cur_pos

                return self.observation, reward, done, info
        else:
            info = {'time_step': self.time_step,
                    'metric_value': self.metric_value,
                    'length': -1
                    }

            self.observation[0] = self.cur_pos

            return self.observation, reward, done, info

    def reset(self):
        self.pipeline = None
        self.nodes = []
        self.time_step = 0
        self.cur_pos = 1
        self.metric_value = 0
        self.observation = np.full(self.pipeline_depth, self.pop)
        self.observation[0] = self.cur_pos

        return self.observation

    def _is_data_operation(self, placed_action):
        if placed_action != self.eop and placed_action != self.pop:
            if any(self.actions_list[0][placed_action] in op for op in self.data_ops):
                return True
        else:
            return False


if __name__ == '__main__':
    file_path_train = 'cases/data/scoring/scoring_train.csv'
    full_path_train = os.path.join(str(fedot_project_root()), file_path_train)

    file_path_test = 'cases/data/scoring/scoring_test.csv'
    full_path_test = os.path.join(str(fedot_project_root()), file_path_test)

    env = PipelineEnv([full_path_train, full_path_test])

    print(env.actions_list)

    for episode in range(1):
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

