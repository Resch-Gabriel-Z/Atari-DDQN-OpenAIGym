hyperparameters = {
    'replay_buffer_size': int(1e6),
    'initial_eps': 1.0,
    'final_eps': 0.01,
    'eps_decay_steps': int(1e6),
    'gamma': 0.99,
    'learning_rate': 0.00025,
    'batch_size': 32,
    'target_update_freq': 10000,
    'number_of_episodes': 10000,
    'max_steps_per_episode': 500,
}
