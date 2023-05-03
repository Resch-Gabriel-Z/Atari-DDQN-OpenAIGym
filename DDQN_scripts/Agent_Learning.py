import numpy as np
import torch

from Replay_Memory import Memory

MIN_SIZE_MEMORY = 1000


def agent_learning(batch_size, gamma, agent, online_net, memory, optimizer, loss_function, device):
    """
    The method described in DQN/DDQN for an agent to learn, this is just an implementation of the algorithm described
    in the paper.
    Args:
        device: if GPU is used, it is passed in here
        batch_size: the number of memories we want to learn
        gamma: the discount factor
        memory: the memory
        agent: the agent
        online_net: the online network
        optimizer: the optimizer
        loss_function: the loss function

    """
    if len(memory) < MIN_SIZE_MEMORY:
        return
    sample_memory = memory.sample(batch_size)
    sample_memory_preprocessed = Memory(*zip(*sample_memory))

    # Get each field of the memory sample
    sample_actions = torch.as_tensor(sample_memory_preprocessed.action, dtype=torch.int64).unsqueeze(-1).to(
        device=device)
    sample_rewards = torch.as_tensor(sample_memory_preprocessed.reward, dtype=torch.float32).unsqueeze(-1).to(
        device=device)
    sample_dones = torch.as_tensor(sample_memory_preprocessed.done, dtype=torch.float32).unsqueeze(-1).to(device=device)
    sample_states = torch.stack(sample_memory_preprocessed.state).to(device=device)
    sample_next_states = torch.as_tensor(np.array(sample_memory_preprocessed.next_state),
                                         dtype=torch.float32).unsqueeze(1).to(device=device)

    # compute the target & q values
    max_q_values_online, _ = torch.max(online_net(sample_next_states), dim=1, keepdim=True)
    target = sample_rewards + gamma * max_q_values_online * (1 - sample_dones)

    q_values = agent.policy_net(sample_states)
    actions_q_values = torch.gather(q_values, dim=1, index=sample_actions)

    # compute loss and perform backpropagation
    loss = loss_function(target, actions_q_values).to(device=device)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
