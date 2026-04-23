import numpy as np
import torch
import time
import os
import json
from env.coop_gridworld import CoopGridWorld
from agents.qmix_agent import QMixAgent

class Args:
    pass

def render(env):
    os.system('cls' if os.name == 'nt' else 'clear')
    grid = np.full((env.height, env.width), '.')
    
    # Obstacle / Wall
    for y in range(env.height):
        if y != env.door_y:
            grid[y, env.wall_col] = 'W'
            
    # Door
    door_open = env.get_door_open()
    grid[env.door_y, env.wall_col] = ' ' if door_open else 'D'
    
    # Plate
    grid[env.plate_pos[1], env.plate_pos[0]] = 'P'
    
    # Treasure
    grid[env.treasure_pos[1], env.treasure_pos[0]] = 'T'
    
    # Agents
    grid[env.agent_pos[0][1], env.agent_pos[0][0]] = 'A'
    grid[env.agent_pos[1][1], env.agent_pos[1][0]] = 'B'
    
    # Print upside down so y=0 is at the bottom
    for y in reversed(range(env.height)):
        print(" ".join(grid[y]))
    print("="*20)

def evaluate():
    args = Args()
    args.n_agents = 2
    args.n_actions = 5
    args.rnn_hidden_dim = 64
    args.mixing_embed_dim = 32
    args.hypernet_layers = 1
    args.lr = 5e-4
    
    env = CoopGridWorld()
    args.obs_shape = env.get_obs_size()
    args.state_shape = env.get_state_size()
    
    agent = QMixAgent(args)
    
    try:
        agent.eval_agent.load_state_dict(torch.load("agent_net.pth"))
        agent.eval_mix.load_state_dict(torch.load("mix_net.pth"))
        print("Models loaded successfully.")
    except FileNotFoundError:
        print("No models found. Please train first.")
        return
        
    obs, state = env.reset()
    hidden_state = agent.init_hidden().expand(args.n_agents, -1)
    
    done = False
    step = 0
    ep_reward = 0
    
    trajectory = []
    
    def log_state():
        trajectory.append({
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
            "door_y": env.door_y
        })

    render(env)
    log_state()
    time.sleep(1)
    
    while not done:
        actions, hidden_state = agent.select_action(obs, hidden_state, epsilon=0.0) # Greedy
        
        reward, done, _ = env.step(actions)
        ep_reward += reward
        obs = env.get_obs()
        
        step += 1
        render(env)
        log_state()
        print(f"Step: {step}, Actions: {actions}, Reward: {reward}")
        time.sleep(0.5)
        
    print(f"Evaluation Complete. Total Reward: {ep_reward}")
    
    # Save trajectory for web visualizer
    with open("web/trajectory.json", "w") as f:
        json.dump(trajectory, f, indent=4)
    print("Trajectory saved to web/trajectory.json")

if __name__ == "__main__":
    evaluate()
