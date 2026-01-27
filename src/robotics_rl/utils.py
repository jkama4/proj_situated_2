import gymnasium as gym

from stable_baselines3 import SAC, TD3


def env_report(
    env: gym.Env
) -> None:
    
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)


def create_sac_agent(
    env: gym.Env,
    lr=1e-3,
    buffer_size=1_000_000,
) -> SAC:
    
    model = SAC(
        policy='MultiInputPolicy',
        env=env,
        learning_rate=lr,
        buffer_size=buffer_size,
        batch_size=256,
        ent_coef='auto',
        gamma=0.99,
        tau=0.005,
        verbose=1
    )

    return model


def create_td3_agent(
    env: gym.Env, 
    lr=1e-3,
    buffer_size=1_000_000,
) -> TD3:
    
    model = TD3(
        policy='MultiInputPolicy',
        env=env,
        learning_rate=lr,
        buffer_size=buffer_size,
        batch_size=256,
        train_freq=(1, 'step'),
        gradient_steps=1,
        policy_delay=2,
        verbose=1
    )

    return model