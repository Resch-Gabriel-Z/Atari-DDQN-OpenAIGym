import cv2
import gym


# Wrapper that takes an Atari game (or any other game with similar observation space) and makes it more suitable for
# training
class AtariWrapper(gym.ObservationWrapper):
    def __int__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space

    def observation(self, observation):
        # We reduce the state to a 84x84 pixel state in Grayscale to improve performance
        observation = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_AREA)
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        return observation
