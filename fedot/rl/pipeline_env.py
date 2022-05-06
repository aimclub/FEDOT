from os import makedirs, walk
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
from fedot.core.utils import fedot_project_root


class EnvironmentDataLoader:
    def __init__(self, path_to_train, path_to_valid, split_ratio=0.7):
        self.split_ratio = split_ratio
        train_datasets = [file_name for (_, _, file_name) in walk(path_to_train)][0]
        valid_datasets = [file_name for (_, _, file_name) in walk(path_to_valid)][0]
        self.train_data, self.val_data = {}, {}

        for dataset in train_datasets:
            path_to_dataset = join(path_to_train, dataset)
            self.train_data[dataset] = InputData.from_csv(path_to_dataset)

        for dataset in valid_datasets:
            path_to_dataset = join(path_to_valid, dataset)
            self.val_data[dataset] = InputData.from_csv(path_to_dataset)

    def get_dataset(self, dataset, mode='train'):
        if mode == 'train':
            return train_test_data_setup(self.train_data[dataset], split_ratio=self.split_ratio)

        elif mode == 'valid':
            return self.train_data[dataset], self.val_data[dataset]


class PipelineGenerationEnvironment(gym.Env):
    metadata = {'render.modes': ['text', 'graph']}

    def __init__(self, dataset_name, dataset_loader, logdir=None, root_length=4, graph_render=True):
        self.dataset_name = dataset_name
        self.dataset_loader = dataset_loader

        self.task_type = TaskTypesEnum.classification
        self.root_length = root_length

        self.actions_list = OperationTypesRepository('all') \
            .suitable_operation(task_type=self.task_type, tags=['reinforce'])

        self.model_ops = OperationTypesRepository('model') \
            .suitable_operation(task_type=self.task_type, tags=['reinforce'])[0]

        self.data_ops = OperationTypesRepository('data_operation') \
            .suitable_operation(task_type=self.task_type, tags=['reinforce'])[0]

        self.actions_list_size = len(self.actions_list[0])
        self.actions_size = self.actions_list_size + 2  # actions list + pop + eop

        self.pop = self.actions_list_size  # pipeline's placeholder
        self.eop = self.actions_list_size + 1  # pipeline's ending

        self.pipeline_idx = 0

        self.nodes = None
        self.pipeline = None
        self.time_step = None
        self.cur_pos = None
        self.metric_value = None
        self.observation = None

        self.action_space = spaces.Discrete(self.actions_size)
        self.observation_space = self.reset()

        if graph_render and logdir is not None:
            self.graph_render_path = join(logdir, 'pipelines')
            if not exists(self.graph_render_path):
                makedirs(self.graph_render_path)

    def step(self, action: int, mode='train'):
        temp_reward = -0.1
        assert self.action_space.contains(action)

        # Штраф для агента за выбор placeholder
        if action == self.pop:
            self._update_observation(action)
            reward, done, info = self._env_response()

            return self.transform_to_one_hot(self.observation), reward, done, info

        # Штраф для агента за выбор окончания построения пайплайна в самом начале
        if action == self.eop and self.nodes == []:
            self._update_observation(action)
            reward, done, info = self._env_response(reward=-0.75)

            return self.transform_to_one_hot(self.observation), reward, done, info

        # # Поощерение, если агент выбрал модель
        # if action != self.eop and self.actions_list[0][action] in self.model_ops:
        #     temp_reward += 0.02
        #
        # # Поощерение, если агент выбрал операцию
        # if action != self.eop and self.actions_list[0][action] in self.data_ops:
        #     temp_reward += 0.01
        #
        #     # Поощерение, если ее поставил в самое начало
        #     if self.cur_pos in [1, 2]:
        #         temp_reward += 0.01

        self._construct_pipeline(action)
        self._update_observation(action)

        # Если агент решил закончить построение пайплайна или достиг лимита,
        # то обороачиваем и проверяем
        if action == self.eop or self.cur_pos == len(self.observation):
            self._final_construct_pipeline()
            self.pipeline = Pipeline(self.nodes[0])
            pipeline = self.pipeline
            try:
                # Проверка пайплайна на ошибки построения
                validate(pipeline, task=Task(self.task_type))

                # Если прошел, то обучаем
                if mode == 'train':
                    train_data, test_data = self.dataset_loader.get_dataset(dataset=self.dataset_name, mode=mode)

                    self.pipeline.fit(train_data)

                    # Проверка полученого пайплайна на тестовых данных
                    results = self.pipeline.predict(test_data)

                    # Подсчитываем метрику
                    self.metric_value = roc_auc(y_true=test_data.target,
                                                y_score=results.predict,
                                                multi_class='ovo',
                                                average='macro')

                elif mode == 'valid':
                    train_data, valid_data = self.dataset_loader.get_dataset(dataset=self.dataset_name, mode=mode)
                    self.pipeline.fit(train_data)

                    results = self.pipeline.predict(valid_data)

                    self.metric_value = roc_auc(y_true=valid_data.target,
                                                y_score=results.predict,
                                                multi_class='ovo',
                                                average='macro')

                    self.render(mode='graph')

                reward = self.metric_value

                _, done, info = self._env_response(length=len(self.nodes), correct=True)
                return self.transform_to_one_hot(self.observation), reward, done, info
            except ValueError:
                reward, done, info = self._env_response(reward=-0.75)
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
        self.observation = np.full(self.root_length + 2, self.pop)
        self.observation[0] = self.cur_pos

        return self.transform_to_one_hot(self.observation)

    def _env_response(self, reward=-1., done=True, length=-1, correct=False):
        info = {
            'time_step': self.time_step,
            'metric_value': self.metric_value,
            'length': length,
            'is_correct': correct,
        }

        return reward, done, info  # reward, done, info

    def _update_observation(self, action):
        self.observation[self.cur_pos] = action
        self.cur_pos += 1
        self.observation[0] = self.cur_pos

    def _construct_pipeline(self, action):
        if action != self.eop:
            if self.time_step == 0:
                self.nodes.append(self.actions_list[0][action])
            else:
                self.nodes.append(PrimaryNode(self.actions_list[0][action]))
            self.time_step += 1

    def _final_construct_pipeline(self):
        if len(self.nodes) == 1:
            self.nodes[0] = PrimaryNode(self.nodes[0])
        else:
            self.nodes[0] = SecondaryNode(self.nodes[0], nodes_from=self.nodes[1:])

    def _is_data_operation(self, placed_action):
        if placed_action != self.eop and placed_action != self.pop:
            if any(self.actions_list[0][placed_action] in op for op in self.data_ops):
                return True
        else:
            return False

    def transform_to_one_hot(self, observation):
        encoded_position = np.eye(self.root_length + 2)[observation[0] - 1]
        encoded_observation = np.array([])

        for obs in observation[1:]:
            encoded_observation = np.append(encoded_observation, np.eye(self.actions_size)[obs])

        output = np.concatenate((encoded_position, encoded_observation.flatten()), axis=None)

        return output

    def render(self, mode='text', plot_in_pycharm=False):
        if mode == 'text':
            output = []
            for primitive in self.observation[2:]:
                if primitive != self.pop and primitive != self.eop:
                    output.append(self.actions_list[0][primitive])
                elif primitive == self.eop:
                    break
                else:
                    output.append('place_holder')

            if self.observation[1] != self.pop and self.observation[1] != self.eop:
                output.append(self.actions_list[0][self.observation[1]])
            elif self.observation[1] == self.eop:
                pass
            else:
                output.append('place_holder')

            print('Pipeline', output)

        elif mode == 'graph':
            if self.pipeline:
                if plot_in_pycharm:
                    self.pipeline.show()
                else:
                    result_path = join(self.graph_render_path, f'pl_{self.dataset_name[:-4]}_{self.pipeline_idx}')
                    self.pipeline.show(path=result_path)
                    self.pipeline_idx += 1


if __name__ == '__main__':
    path_to_train = join(fedot_project_root(), 'fedot/rl/data/train/')
    path_to_valid = join(fedot_project_root(), 'fedot/rl/data/valid/')

    edl = EnvironmentDataLoader(path_to_train, path_to_valid)
    env = PipelineGenerationEnvironment(dataset_name='adult.csv', dataset_loader=edl)

    print(env.actions_list[0])
    print([i for i in range(len(env.actions_list[0]))])

    for episode in range(5):
        iter = 0
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            print('episode:', episode)
            print('iteration:', iter)
            print('state', state)
            action = int(input())
            state, reward, done, info = env.step(action)
            print('reward: %6.2f' % reward)
            total_reward += reward

            env.render(mode='text')
            env.render(mode='graph', plot_in_pycharm=True)

            if done:
                print('episode:', episode, 'state', state)
                print('Done')

            iter += 1
