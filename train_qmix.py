import argparse
import numpy as np
import torch
import random
import json
import os
from env.coop_gridworld import CoopGridWorld
from agents.qmix_agent import QMixAgent
from agents.replay_buffer import ReplayBuffer

class Args:
    pass

def train(n_episodes: int = 10000):
    args = Args()
    args.n_agents = 2
    args.n_actions = 5
    args.obs_shape = 5
    args.state_shape = 5
    args.rnn_hidden_dim = 64
    args.mixing_embed_dim = 32
    args.hypernet_layers = 1
    args.lr = 5e-4
    args.gamma = 0.99
    args.grad_norm_clip = 10.0
    args.target_update_interval = 200 # episodes
    
    # n_episodes is passed as a parameter (set via CLI argument)
    batch_size = 32
    buffer_capacity = 10000
    max_ep_len = 25
    
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_anneal_time = 5000
    
    # How often to save an episode trajectory for visualization
    save_episode_interval = 50
    
    env = CoopGridWorld()
    args.obs_shape = env.get_obs_size()
    args.state_shape = env.get_state_size()
    
    agent = QMixAgent(args)
    buffer = ReplayBuffer(buffer_capacity, max_ep_len, args)
    
    # ── Resume from saved checkpoint if it exists ──────────────────────────
    os.makedirs("web", exist_ok=True)
    episode_offset = 0      # how many episodes were already trained
    training_episodes = []  # will be populated from existing data or start fresh
    ep_rewards = []

    # ── Fresh training: ignore any existing weights ──────────────────────────
    print("Fresh training from scratch — ignoring any saved weights.")

    # Always start fresh — reset trajectory data
    training_episodes = []
    episode_offset = 0
    epsilon = epsilon_start
    print(f"Fresh training with epsilon = {epsilon:.4f}")
    
    print(f"Starting {n_episodes} additional training episodes...")

    for ep in range(n_episodes):
        obs, state = env.reset()
        hidden_state = agent.init_hidden().expand(args.n_agents, -1)
        
        episode = {
            "states": [state],
            "obs": [obs],
            "actions": [],
            "rewards": [],
            "dones": []
        }
        
        # Record trajectory for this episode
        ep_trajectory = []
        ep_actions_log = []
        
        ep_reward = 0
        done = False
        step = 0
        
        # Log initial state
        ep_trajectory.append({
            "step": 0,
            "agent_0": env.agent_pos[0].tolist(),
            "agent_1": env.agent_pos[1].tolist(),
            "door_open": bool(env.get_door_open()),
            "plate_pos": env.plate_pos.tolist(),
            "treasure_pos": env.treasure_pos.tolist(),
            "reward": 0.0,
            "width": env.width,
            "height": env.height,
            "wall_col": env.wall_col,
            "door_y": env.door_y
        })
        
        while not done:
            actions, hidden_state = agent.select_action(obs, hidden_state, epsilon)
            
            reward, done, _ = env.step(actions)
            ep_reward += reward
            
            next_obs = env.get_obs()
            next_state = env.get_state()
            
            episode["actions"].append([[a] for a in actions])
            episode["rewards"].append([reward])
            episode["dones"].append([1 if done else 0])
            
            episode["states"].append(next_state)
            episode["obs"].append(next_obs)
            
            step += 1
            
            # Log step for trajectory
            ep_trajectory.append({
                "step": step,
                "agent_0": env.agent_pos[0].tolist(),
                "agent_1": env.agent_pos[1].tolist(),
                "door_open": bool(env.get_door_open()),
                "plate_pos": env.plate_pos.tolist(),
                "treasure_pos": env.treasure_pos.tolist(),
                "reward": float(ep_reward),
                "width": env.width,
                "height": env.height,
                "wall_col": env.wall_col,
                "door_y": env.door_y,
                "actions": actions
            })
            
            obs = next_obs
            
        # Anneal epsilon
        if epsilon > epsilon_end:
            epsilon -= (epsilon_start - epsilon_end) / epsilon_anneal_time
            
        # Pad episode to max_ep_len
        actual_len = len(episode["actions"])
        if actual_len < max_ep_len:
            pad_len = max_ep_len - actual_len
            episode["actions"].extend([[[0], [0]]] * pad_len)
            episode["rewards"].extend([[0.0]] * pad_len)
            episode["dones"].extend([[1.0]] * pad_len) # After done, dones are 1
            episode["states"].extend([episode["states"][-1]] * pad_len)
            episode["obs"].extend([episode["obs"][-1]] * pad_len)
            
        buffer.push(episode)
        ep_rewards.append(ep_reward)
        
        # Track training metrics for every episode
        loss_val = None
        
        # Train
        if len(buffer) >= batch_size:
            batch = buffer.sample(batch_size)
            loss_val = agent.train(batch, max_ep_len)
            
        # Update target
        if ep % args.target_update_interval == 0:
            agent.update_targets()
        
        # Save episode trajectory periodically
        should_save = (ep % save_episode_interval == 0) or (ep >= n_episodes - 10)
        if should_save:
            success = any(np.array_equal(pos, env.treasure_pos) for pos in env.agent_pos)
            avg_r = float(np.mean(ep_rewards[-500:])) if ep_rewards else 0.0
            
            training_episodes.append({
                "episode": ep,
                "epsilon": round(float(epsilon), 4),
                "total_reward": round(float(ep_reward), 2),
                "steps": actual_len,
                "success": bool(success),
                "avg_reward_500": round(avg_r, 2),
                "loss": round(float(loss_val), 6) if loss_val is not None else None,
                "trajectory": ep_trajectory
            })
            
            # Save to file
            os.makedirs("web", exist_ok=True)
            with open("web/training_data.json", "w") as f:
                json.dump({
                    "total_episodes": ep + 1,
                    "n_episodes_target": n_episodes,
                    "episodes": training_episodes
                }, f, indent=2)
            
        if ep % 100 == 0:
            avg_reward = np.mean(ep_rewards[-100:])
            print(f"Episode: {ep}, Epsilon: {epsilon:.2f}, Avg Reward (last 100): {avg_reward:.2f}, Episodes saved: {len(training_episodes)}")

    print("Training Complete. Saving models...")
    torch.save(agent.eval_agent.state_dict(), "agent_net.pth")
    torch.save(agent.eval_mix.state_dict(), "mix_net.pth")
    print(f"Models saved. {len(training_episodes)} total episode trajectories in web/training_data.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement QMIX – Multi-Agent RL")
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=10000,
        help="Nombre d'épisodes d'entraînement (défaut : 10000)"
    )
    cli_args = parser.parse_args()
    print(f"▶ Lancement de l'entraînement pour {cli_args.n_episodes} épisode(s)...")
    train(n_episodes=cli_args.n_episodes)
