import gymnasium as gym

def env_report(
    env: gym.Env
) -> None:
    
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)