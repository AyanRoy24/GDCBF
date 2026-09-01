import numpy as np
from PIL import Image
from typing import Dict
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
pygame.init()
from env.rl_compat import gym
import numpy as np
import jax.numpy as jnp
import time
from tqdm.auto import trange  # noqa

def check_coverage(barrier_values, threshold=0.0):
    """
    Fraction of states the CBF certifies as safe.
    Since h(s,a) = -c(s,a) and Q_h ≤ 0 means safe, use <= 0
    """
    return np.mean(barrier_values <= threshold)

def check_valid(barrier_values, next_barrier_values, alpha=0.1):
    """
    Fraction of transitions satisfying the discrete-time CBF condition.
    CBF condition: h(x_{t+1}) - h(x_t) + α*h(x_t) >= 0
    Rearranged: h(x_{t+1}) >= (1-α)*h(x_t)
    """
    one_minus_alpha = 1 - alpha
    # For barrier values <= 0 (safe), condition is automatically satisfied
    # For barrier values > 0 (unsafe), check the CBF condition
    safe_mask = (barrier_values <= 0)
    unsafe_mask = (barrier_values > 0)
    
    # CBF condition for unsafe states
    cbf_condition = next_barrier_values >= one_minus_alpha * barrier_values
    
    # Valid if: (safe) OR (unsafe AND CBF condition satisfied)
    valid = safe_mask | (unsafe_mask & cbf_condition)
    return np.mean(valid)


def _surface_to_pil(frame):
    """Convert pygame.Surface or numpy array to PIL.Image."""
    if isinstance(frame, pygame.Surface):
        arr = pygame.surfarray.array3d(frame)  # shape (w, h, 3)
        arr = np.transpose(arr, (1, 0, 2))     # -> (h, w, 3)
        return Image.fromarray(arr.copy())
    if isinstance(frame, np.ndarray):
        return Image.fromarray(frame)
    raise TypeError(f"Unsupported frame type: {type(frame)}")

def evaluate_md(obs_mean, obs_std, seed, env_id,  eval_num, agent, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False) -> Dict[str, float]:
    episode_rets, episode_costs, episode_lens = [], [], []
    frames_all = []
    for ep_idx in trange(num_episodes, desc="Evaluating", leave=False):
        obs, info = env.reset()
        if obs_mean is not None and obs_std is not None:
            obs = (obs - obs_mean) / (obs_std)
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0

        # collect frames for this episode
        frames = []

        while True:
            if render:
                frame = env.render(mode="topdown", 
                                   scaling=6, 
                                   window=False,
                                   camera_position=(50, -50),
                                   screen_size=(300, 700), #(w,h)
                                   screen_record=True,
                                   draw_target_vehicle_trajectory=True)
                # convert immediately to PIL.Image (safe) and store
                try:
                    pil_img = _surface_to_pil(frame)
                except TypeError:
                    # fallback: if env returns something else, try Image.fromarray directly
                    pil_img = Image.fromarray(np.array(frame))

                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(pil_img)
                label_text = f"Episode {ep_idx + 1}/{num_episodes}"
                try:
                    # Try to use a larger font if available
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
                except:
                    # Fallback to default font
                    font = ImageFont.load_default()
                
                # Draw text with background for better visibility
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Position: top-left corner with padding
                x, y = 10, 10
                # Draw background rectangle
                draw.rectangle([x-5, y-5, x+text_width+5, y+text_height+5], fill=(0, 0, 0, 180))
                # Draw text
                draw.text((x, y), label_text, fill=(255, 255, 255), font=font)

                frames.append(pil_img)
                time.sleep(1e-3)
                
            action, agent = agent.eval_actions(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            if obs_mean is not None and obs_std is not None:
                next_obs_norm = (next_obs - obs_mean) / (obs_std)
            else:
                next_obs_norm = next_obs
            
            cost = info["cost"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            obs = next_obs_norm
            
            if terminated or truncated:
                break
                
        
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

        if len(frames) > 0:
            frames_all.extend(frames)
        

    if len(frames_all) > 0:
        try:
            pil_images = []
            for img in frames_all:
                if isinstance(img, Image.Image):
                    pil_images.append(img.convert("RGBA"))
                elif isinstance(img, np.ndarray):
                    pil_images.append(Image.fromarray(img).convert("RGBA"))
                else:
                    pil_images.append(Image.fromarray(np.array(img)).convert("RGBA"))

            # Convert to palette mode suitable for GIF
            paletted = [im.convert("P", palette=Image.ADAPTIVE) for im in pil_images]
            out_name = f"{env_id}_{seed}.gif"
            paletted[0].save(
                out_name,
                format="GIF",
                save_all=True,
                append_images=paletted[1:],
                duration=50,
                loop=0,
            )
            print(f"Saved concatenated GIF: {out_name}")
        except Exception as e:
            print("Failed to save concatenated GIF:", e)

    return {
        "return": np.mean(episode_rets),
        "cost": np.mean(episode_costs),
        "episode_len": np.mean(episode_lens),
    }

def evaluate(obs_mean, obs_std, agent, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False) -> Dict[str, float]:
    episode_rets, episode_costs, episode_lens = [], [], []
    barriers, next_barriers = [], []
    
    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs, info = env.reset()
        if obs_mean is not None and obs_std is not None:
            obs = (obs - obs_mean) / (obs_std)
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        
        while True:
            if render:
                env.render()
                time.sleep(1e-3)
                
            action, agent = agent.eval_actions(jnp.array(obs))
            action = np.array(action)            
            barrier_value = agent.barrier_values(jnp.expand_dims(obs, axis=0)).item()
            barriers.append(barrier_value)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            if obs_mean is not None and obs_std is not None:
                next_obs = (next_obs - obs_mean) / (obs_std)
            
            next_barrier_value = agent.barrier_values(jnp.expand_dims(next_obs, axis=0)).item()
            next_barriers.append(next_barrier_value)
            
            cost = info["cost"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            obs = next_obs
            
            if terminated or truncated:
                break
                
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)
    
    barriers = jnp.array(barriers)
    next_barriers = jnp.array(next_barriers)
    
    validity = check_valid(barriers, next_barriers, alpha=0.9)
    
    coverage = check_coverage(barriers, threshold=0.0)

    return {
        "return": np.mean(episode_rets),
        "cost": np.mean(episode_costs),
        "episode_len": np.mean(episode_lens),
        "coverage": coverage,
        "validity": validity,
    }
    

