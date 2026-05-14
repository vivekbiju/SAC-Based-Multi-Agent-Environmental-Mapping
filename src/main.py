import torch
import os
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter
from src.environment import MultiAgentMonitoring
from src.agent import SACAgent
from src.utils import evaluate_agent  # Ensure this function is in your utils.py

# Professional configuration for CV
cfg = edict({
    "map_size": (58, 41),
    "num_agents": 2,
    "map_data_path": "data/ypacarai_map.csv",
    "collision_threshold": 0.05,
    "collision_penalty": -2.0,
    "reward_visited_cells": 1.0,
    "max_steps": 200,
    "hidden_size": 256,
    "lr": 3e-4,
    "replay_size": 1000000,
    "batch_size": 64,
    "gamma": 0.99,
    "tau": 0.005,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
})

def train():
    # 1. Create directories explicitly
    log_dir = os.path.join("runs_tuning", "sac_experiment")
    os.makedirs(log_dir, exist_ok=True)
    
    env = MultiAgentMonitoring(cfg)
    agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0] * 2, cfg)
    
    # 2. Initialize TensorBoard Writer
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Logging to: {log_dir}")

    for ep in range(100):
        obs, _ = env.reset()
        episode_reward = 0
        
        for step in range(cfg.max_steps):
            action = agent.select_action(obs)
            next_obs, reward, _, truncated, info = env.step(action.reshape(cfg.num_agents, 2))
            
            agent.memory.append((obs, action, reward, next_obs, truncated))
            agent.update() # Update networks
            
            obs = next_obs
            episode_reward += reward
            if truncated: break

        # 3. Log to TensorBoard and Flush
        writer.add_scalar('Episode/Reward', episode_reward, ep)
        if ep % 5 == 0:
            writer.flush()  # Forces data to appear in TensorBoard immediately
            print(f"Episode {ep} | Reward: {episode_reward:.2f}")

        # 4. Periodic Evaluation (Saves .png maps)
        if ep % 50 == 0 and ep > 0:
            print(f"--- Running Periodic Evaluation at Episode {ep} ---")
            evaluate_agent(env, agent, num_episodes=1, seed=cfg.seed, log_dir=log_dir, run_name=f"eval_ep{ep}")

    # Final cleanup
    writer.close()
    torch.save(agent.policy.state_dict(), os.path.join(log_dir, "final_model.pth"))

if __name__ == "__main__":
    train()