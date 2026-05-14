import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import LinearNDInterpolator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_agent(env, agent, num_episodes, seed, log_dir, run_name):
    """
    Performs deterministic evaluation and generates reconstructed maps.
    This is the core for your MSc Thesis results.
    """
    all_rewards = []
    # Use base environment to access map data
    base_env = env.env if hasattr(env, 'env') else env
    
    # Storage for sensing data across the evaluation
    all_coords = []
    all_values = []

    for i in range(num_episodes):
        obs, info = env.reset(seed=seed + i)
        done = False
        truncated = False
        ep_reward = 0

        while not (done or truncated):
            # Evaluate=True uses deterministic actions (mean)
            action = agent.select_action(obs, eval=True)
            obs, reward, done, truncated, info = env.step(action.reshape(base_env.num_agents, 2))
            ep_reward += reward

            # Collect sensed data from info dict
            # Expected info['sensed_values'] format: [(x, y, value), ...]
            if 'sensed_values' in info:
                for x, y, val in info['sensed_values']:
                    all_coords.append([x, y])
                    all_values.append(val)
        
        all_rewards.append(ep_reward)

    coords = np.array(all_coords)
    values = np.array(all_values)

    # Visualization and Metric Calculation
    if len(coords) > 3:
        try:
            # Create a grid for the entire map
            grid_y, grid_x = np.mgrid[0:base_env.map_size[0], 0:base_env.map_size[1]]
            
            # Linear Interpolation to reconstruct the map
            interp = LinearNDInterpolator(coords, values)
            est_map = interp(grid_x, grid_y)

            # Calculate Metrics (Ignoring NaNs where the agent didn't explore)
            mask = ~np.isnan(est_map)
            if np.any(mask):
                gt_flat = base_env.gt_map_normalized[mask]
                est_flat = est_map[mask]
                
                mae = mean_absolute_error(gt_flat, est_flat)
                r2 = r2_score(gt_flat, est_flat)
                
                print(f"Evaluation {run_name} | MAE: {mae:.4f} | R2: {r2:.4f}")

                # Save the Visual Comparison
                _save_comparison_plot(base_env.gt_map_normalized, est_map, coords, log_dir, run_name)
        except Exception as e:
            print(f"Interpolation failed: {e}")

    return np.mean(all_rewards)

def _save_comparison_plot(gt, est, coords, log_dir, name):
    """Internal helper to save PNG plots for your thesis/CV."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Ground Truth
    im0 = axes[0].imshow(gt, origin='lower', cmap='viridis')
    axes[0].set_title("Ground Truth Map")
    plt.colorbar(im0, ax=axes[0])
    
    # Plot Reconstruction + Agent Path (Sensed Points)
    im1 = axes[1].imshow(est, origin='lower', cmap='viridis')
    axes[1].scatter(coords[:, 0], coords[:, 1], c='red', s=1, alpha=0.5, label='Agent Path')
    axes[1].set_title("Reconstructed Map (Sensing)")
    plt.colorbar(im1, ax=axes[1])
    
    plot_path = os.path.join(log_dir, f"{name}.png")
    plt.savefig(plot_path)
    plt.close()