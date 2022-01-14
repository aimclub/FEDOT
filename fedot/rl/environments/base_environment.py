from fedot.rl.environments.linear_pipeline_generator import LinearPipelineGenerator


class BaseEnvironment(object):
    def __init__(self):
        self.env = LinearPipelineGenerator()
        self.steps_taken = 0
        self.reward = self.get_reward()

    def get_actions(self):
        """ Позволяет агенту запросить набор действий, которые он может выполнить"""
        return self.env.get_actions()

    def is_done(self):
        """ Проверка об окончании генерации """
        return self.steps_taken == 10

    def reset(self):
        """ Обновление среды """
        self.steps_taken = 0

        return self.env.reset()

    def step(self, action):
        """ Приминение действия в среде """
        if self.is_done():
            raise Exception('Pipeline generation is over')

        self.steps_taken += 1

        reward = self.reward()

        return next_state, reward, done, {}

    def get_reward(self):
        return 0
