import random
from collections import namedtuple, deque

Memory = namedtuple('Memory', ('state', 'action', 'done', 'next_state', 'reward'))


class ReplayMemory(object):
    """
    A class to save memories in an effective, easily callable named tuple
    """

    def __init__(self, capacity):
        """
        initialization of the memory with a capacity. the deque makes sure it deletes the oldest entries
        Args:
            capacity: number of memories we want to hold on
        """
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """
        A method to save new memories into the deque
        Args:
            *args: all the arguments we want to save in the memory (typically the (next-)state, reward, action and done)
        """
        self.memory.append(Memory(*args))

    def sample(self, batch_size):
        """
        A method to sample random memories from the named tuple
        Args:
            batch_size: the number of memories we want to sample

        Returns:
            random memories as named tuples
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        A method to return the length of the memory (at most the capacity
        Returns:
            the length
        """
        return len(self.memory)
