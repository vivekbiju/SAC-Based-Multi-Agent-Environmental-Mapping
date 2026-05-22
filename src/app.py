import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from easydict import EasyDict as edict
from scipy.interpolate import LinearNDInterpolator

# Core directory module imports
from src.environment import MultiAgentMonitoring
from src.agent import SACAgent

# --- Page Setup & Typography Layout ---
st.set_page_config(page_title="SAC Path Planning Dashboard", layout="wide")
st.title("🤖 Autonomous Multi-Agent Environmental Monitoring Dashboard")
st.markdown(
    "Developed as part of an MSc Robotics Thesis. This system runs a trained **Soft Actor-Critic (SAC)** "
    "reinforcement learning policy to optimize spatial coverage and map continuous tracking targets."
)

# --- Sidebar Controls Layout Panel ---
st.sidebar.header("🕹️ Simulation Parameters")
map_choice = st.sidebar.selectbox("Target Map Environment", ["Ypacarai Lake (Dynamic Dimensions)"])
max_steps = st.sidebar.slider("Mission Step Limit", 50, 500, 150)
animation_speed = st.sidebar.slider("Render Delay Frame-rate (Seconds)", 0.01, 0.5, 0.05)

# --- Baseline Framework Settings Configuration ---
cfg = edict({
    "map_size": (58, 41),  # This gets dynamically overwritten on initialization below
    "num_agents": 2,
    "map_data_path": "data/ypacarai_map.csv",
    "collision_threshold": 0.05,
    "collision_penalty": -2.0,
    "reward_visited_cells": 1.0,
    "max_steps": max_steps,
    "hidden_size": 256,
    "lr": 3e-4,
    "replay_size": 1000,
    "batch_size": 64,
    "gamma": 0.99,
    "tau": 0.005,
    "device": "cpu"
})

# --- Main Core Execution Pipeline Engine ---
if st.sidebar.button("🚀 Launch Autonomous Mission"):
    
    # 1. Initialize Environment
    env = MultiAgentMonitoring(cfg)
    
    # 2. DYNAMIC MAP DIMENSION FIX: Extract exact dimensions directly from the CSV matrix data
    true_y_pixels = env.gt_map_normalized.shape[0]  # Total Row Heights (Y-axis)
    true_x_pixels = env.gt_map_normalized.shape[1]  # Total Column Widths (X-axis)
    cfg.map_size = (true_y_pixels, true_x_pixels)   # Update config parameters seamlessly
    
    # 3. Instantiate Policy and Dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] * 2
    agent = SACAgent(state_dim, action_dim, cfg)
    
    # 4. Attempt Brain Parameter Load Checks
    model_path = "runs_tuning/sac_experiment/final_model.pth"
    if os.path.exists(model_path):
        agent.policy.load_state_dict(torch.load(model_path, map_location="cpu"))
        st.success("💾 Pre-trained SAC neural parameters verified and safely loaded.")
    else:
        st.warning("⚠️ Weights checkpoints missing. Falling back to active random exploration arrays initialization.")

    # 5. Build Container Display Windows Block
    status_text = st.empty()
    progress_bar = st.progress(0)
    plot_placeholder = st.empty()

    # 6. Reset Matrix Environments Tracker Registers
    obs, info = env.reset(seed=42)
    all_coords = []
    all_values = []
    
    # Warm initialization tracking loop with starting positions data
    if 'sensed_values' in info:
        for x, y, val in info['sensed_values']:
            all_coords.append([x, y])
            all_values.append(val)
            
    # 7. Operational Simulation Evaluation Step Loop
    for step in range(max_steps):
        action = agent.select_action(obs, eval=True)
        obs, reward, _, done, info = env.step(action.reshape(cfg.num_agents, 2))
        
        if 'sensed_values' in info:
            for x, y, val in info['sensed_values']:
                all_coords.append([x, y])
                all_values.append(val)
        
        coords_arr = np.array(all_coords)
        
        # 8. Render Engine Pipeline Block
        if len(coords_arr) > 3:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Subplot 1: Ground Truth Visual Canvas
            im0 = axes[0].imshow(env.gt_map_normalized, origin='lower', cmap='viridis', aspect='equal')
            axes[0].set_title("Ground Truth Baseline Map", fontsize=12)
            axes[0].set_xlabel("X Space Coordinate")
            axes[0].set_ylabel("Y Space Coordinate")
            axes[0].set_xlim(0, true_x_pixels)
            axes[0].set_ylim(0, true_y_pixels)
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            
            # Subplot 2: Corrected Map Interpolation Array Rebuilding Grid
            try:
                # FIX: Build interpolation meshes perfectly aligned to exact matrix shape arrays
                grid_y, grid_x = np.mgrid[0:true_y_pixels, 0:true_x_pixels]
                
                interp = LinearNDInterpolator(coords_arr, np.array(all_values))
                est_map = interp(grid_x, grid_y)
                
                # Match baseline coordinate conventions precisely
                im1 = axes[1].imshow(est_map, origin='lower', cmap='viridis', aspect='equal')
                axes[1].scatter(coords_arr[:, 0], coords_arr[:, 1], c='red', s=8, alpha=0.7, label='Agent Steps')
                axes[1].set_title(f"Live Reconstructed Estimation Matrix (Step: {step})", fontsize=12)
                axes[1].set_xlabel("X Space Coordinate")
                axes[1].set_ylabel("Y Space Coordinate")
                
                # Dynamic scale matching fixes
                axes[1].set_xlim(0, true_x_pixels)
                axes[1].set_ylim(0, true_y_pixels)
                fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
                
            except Exception as e:
                axes[1].text(0.5, 0.5, "Processing matrix triangulation...", ha='center', va='center')
            
            # Force replace previous frame securely inside layout slot area container
            with plot_placeholder.container():
                st.pyplot(fig)
            plt.close(fig)  # Memory clear cleanup routines

        # Telemetry update monitoring fields refreshing string injections
        status_text.markdown(f"### ⚡ Telemetry Monitor — Frame Step: **{step}/{max_steps}** | Aggregated Sensed Coordinates: **{len(all_coords)}**")
        progress_bar.progress((step + 1) / max_steps)
        
        time.sleep(animation_speed)
        
    st.balloons()
    st.success("🎯 Spatial exploration tracking route completed successfully.")