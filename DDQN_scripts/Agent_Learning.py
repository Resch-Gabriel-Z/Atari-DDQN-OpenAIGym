import numpy as np
import torch

from Replay_Memory import Memory


def agent_learning(batch_size, gamma, **kwargs):
    memory_l = kwargs['memory']
    agent_l = kwargs['agent']
    online_l = kwargs['online_network']
    optimizer_l = kwargs['optimizer']
    loss_func_l = kwargs['loss_function']

    # Learn only if the memory has enough experience to learn from
    if len(memory_l) < 1000:
        return
    # Sample the memory
    sample_memory = memory_l.sample(batch_size)
    sample_memory_preprocessed = Memory(*zip(*sample_memory))

    # Get each field of the memory sample
    sample_actions = torch.tensor(sample_memory_preprocessed.action, dtype=torch.int64).unsqueeze(-1)
    sample_rewards = torch.tensor(sample_memory_preprocessed.reward, dtype=torch.float32)
    sample_dones = torch.tensor(sample_memory_preprocessed.done, dtype=torch.float32)
    sample_states = torch.stack(sample_memory_preprocessed.state)
    sample_next_states = torch.tensor(np.array(sample_memory_preprocessed.next_state), dtype=torch.float32).unsqueeze(1)

    # compute the target & q values
    max_q_values_online, _ = torch.max(online_l(sample_next_states), dim=1)
    target = sample_rewards + gamma * max_q_values_online * (1 - sample_dones)

    q_values = agent_l.policy_net(sample_states)
    actions_q_values = torch.gather(q_values, dim=1, index=sample_actions)

    # compute loss and perform backpropagation
    loss = loss_func_l(target.unsqueeze(1), actions_q_values)
    optimizer_l.zero_grad()
    loss.backward()
    optimizer_l.step()
