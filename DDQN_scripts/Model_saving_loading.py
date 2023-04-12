import os

import torch


def load_model_dict(path, name, **kwargs):
    policy_net_load = kwargs['policy_state_dict']
    online_net_load = kwargs['online_state_dict']
    optimizer_load = kwargs['optimizer_state_dict']
    start_load = kwargs['start']
    total_steps_load = kwargs['total_steps']
    memory_load = kwargs['memory_savestate']

    # Basic function to look for a file and then loads it variables taken from a dict
    if os.path.exists(path + '/' + name + '.pth'):
        print('Save File Found!')
        checkpoint = torch.load(path + '/' + name + '.pth')

        policy_net_load.load_state_dict(checkpoint['policy_state_dict'])
        online_net_load.load_state_dict(checkpoint['online_state_dict'])
        start_load = checkpoint['start'] + 1
        optimizer_load.load_state_dict(checkpoint['optimizer_state_dict'])
        total_steps_load = checkpoint['total_steps']
        memory_load = checkpoint['memory_savestate']
    else:
        print('No Save File Found. Begin new training')

    return start_load, total_steps_load, memory_load


def save_model_dict(path, name, **kwargs):
    policy_net_save = kwargs['policy_state_dict']
    online_net_save = kwargs['online_state_dict']
    optimizer_save = kwargs['optimizer_state_dict']
    start_save = kwargs['start']
    total_steps_save = kwargs['total_steps']
    memory_save = kwargs['memory_savestate']

    # Basic torch function that saves variables as a dict to load them afterward
    torch.save({
        'policy_state_dict': policy_net_save.state_dict(),
        'online_state_dict': online_net_save.state_dict(),
        'optimizer_state_dict': optimizer_save.state_dict(),
        'start': start_save,
        'total_steps': total_steps_save,
        'memory_savestate': memory_save,
    }, path + '/' + name + '.pth')


# Saving a model after training is completed (only the policy state dict to load it into the Agent)
def save_final_model(path, name, **kwargs):
    final_policy = kwargs['policy_state_dict']
    torch.save({'policy_state_dict': final_policy.state_dict()}, path + '/' + name + '.pt')
