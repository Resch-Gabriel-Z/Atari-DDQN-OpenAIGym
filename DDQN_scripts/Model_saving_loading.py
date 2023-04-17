import os

import torch


def load_model_dict(path, name, policy_net, online_net, optimizer, starting_point, episode_reward_tracker, total_steps,
                    memory):
    """
    A Method that loads certain variables I decided to be valuable from a dictionary.
    Args:
        path: the path we look for the checkpoint
        name: the name of the file
        policy_net: the policy parameters
        online_net: the online network parameters
        optimizer: the optimizer parameters
        starting_point: the starting point we want to learn on (in case we want to divide the training session)
        episode_reward_tracker: the rewards over all episodes
        total_steps: the total steps done so far
        memory: the memory we have currently

    Returns:
        some variables can be directly loaded from the directory, others have to be returned, these are the ones.

    """
    if os.path.exists(f'{path}/{name}.pt'):
        print('Save File Found!')
        checkpoint = torch.load(f'{path}/{name}.pt')

        policy_net.load_state_dict(checkpoint['policy_state_dict'])
        online_net.load_state_dict(checkpoint['online_state_dict'])
        starting_point = checkpoint['start'] + 1
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        total_steps = checkpoint['total_steps']
        memory = checkpoint['memory_savestate']
        episode_reward_tracker = checkpoint['episode_reward_tracker']
    else:
        print('No Save File Found. Begin new training')

    return starting_point, episode_reward_tracker, total_steps, memory


def save_model_dict(path, name, policy_net, online_net, optimizer, starting_point, episode_reward_tracker, total_steps,
                    memory):
    """
    A simple method to save the model after an episode
    Args:
        path: the path we will save the checkpoint in
        name: the name of the file
        policy_net: the policy parameters
        online_net: the online network parameters
        optimizer: the optimizer parameters
        starting_point: the starting point we want to learn on (in case we want to divide the training session)
        episode_reward_tracker: the rewards over all episodes
        total_steps: the total steps done so far
        memory: the memory we have currently
    """
    torch.save({
        'policy_state_dict': policy_net.state_dict(),
        'online_state_dict': online_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'start': starting_point,
        'total_steps': total_steps,
        'memory_savestate': memory,
        'episode_reward_tracker': episode_reward_tracker,
    }, f'{path}/{name}.pt')


def save_final_model(path, name, model):
    """
    A simple method to save the parameters after we trained the model
    Args:
        path: the path to save the final model
        name: the name of the final model.
        model: the model itself
    """
    torch.save(model.state_dict(), f'{path}/{name}.pt')
