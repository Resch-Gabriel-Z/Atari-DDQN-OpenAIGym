import cv2
import gym

SCREEN_SIZE = 84


class AtariWrapper(gym.ObservationWrapper):
    """
    A Wrapper class that inherits from the gym Observation Wrapper to preprocess any state in the Atari game.
    To be exact, similar to the paper, we have a 84x84 gray scale picture.
    """
    def __int__(self, env):
        """
        Args:
            env: The environment to call it if necessary.
        """
        super().__init__(env)
        self.observation_space = env.observation_space

    def observation(self, observation):
        """
        Here we preprocess each state as written above. We use the cv2 library for that.
        Args:
            observation: the current state

        Returns:
            the preprocessed state
        """
        observation = cv2.resize(observation, (SCREEN_SIZE, SCREEN_SIZE), interpolation=cv2.INTER_AREA)
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        return observation


def environment_maker(game, render_mode='rgb_array'):
    """
    A function that we use to actually create the wrapped environment
    Args:
        game: the ID of the game.
        render_mode: the render mode as in the original environment

    Returns:
        the wrapped environment
    """
    env_base = gym.make(game, render_mode=render_mode)
    env_wrapped = AtariWrapper(env_base)
    return env_wrapped
