import os
import torch
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import time

# IMPORT TQC FROM CONTRIB
from sb3_contrib import TQC
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# TQC uses entropy for exploration, so we do not need NormalActionNoise

device = "cuda" if torch.cuda.is_available() else "cpu"

save_path = "logs/fetch_pick_place_her_tqc"
os.makedirs(save_path, exist_ok=True)

gym.register_envs(gymnasium_robotics)

if __name__ == "__main__":
    print(f"--- HARDWARE CHECK ---")
    print(f"Device: {device.upper()}")
    print(f"CPU Cores Available: {os.cpu_count()}")
    print(f"----------------------")

    n_envs = 16
    
    print(f"Spawning {n_envs} parallel environments via SubprocVecEnv...")
    
    train_env = make_vec_env(
        "FetchPickAndPlace-v4", 
        n_envs=n_envs, 
        vec_env_cls=SubprocVecEnv
    )
    
    train_env = VecMonitor(train_env, filename=f"{save_path}/train_monitor.csv")

    eval_env = gym.make("FetchPickAndPlace-v4")
    eval_env = Monitor(eval_env, filename=f"{save_path}/eval_monitor.csv")



    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=2500,          
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=6250,
        save_path=save_path,
        name_prefix="tqc_checkpoint"
    )

    print("Initializing TQC + HER with RTX 4090 Optimizations...")
    
    model = TQC(
        "MultiInputPolicy",
        train_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        verbose=1,
        learning_rate=3e-4,
        
        gamma=0.99,
        tau=0.005,
        
        batch_size=1024,
        buffer_size=1_000_000,
        
        # TQC Specific: n_critics is usually 2, quantiles logic is internal
        policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),

        train_freq=1,
        gradient_steps=-1,
        learning_starts=5000,
        device=device
    )

    # ---------------------------------------------------------
    # 6. TRAINING LOOP
    # ---------------------------------------------------------
    total_timesteps = 1_000_000
    print(f"Starting training for {total_timesteps} timesteps...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps, 
        callback=[eval_callback, checkpoint_callback]
    )
    
    end_time = time.time()
    total_time_min = (end_time - start_time) / 60
    print(f"Training Finished in {total_time_min:.2f} minutes.")

    # Final Save
    final_path = os.path.join(save_path, "final_model_pick_place.zip")
    model.save(final_path)
    print(f"Model saved to {final_path}")
    
    # Close processes
    train_env.close()
    eval_env.close()