import cv2
import gym


class Atari_wrapper(gym.ObservationWrapper):
    def __int__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space

    def observation(self, observation):
        observation = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_AREA)
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        return observation
