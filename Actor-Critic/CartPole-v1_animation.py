import gym
import time

env = gym.make("CartPole-v1", render_mode="human")  # 'human' to open a window
obs = env.reset()[0]  # new Gym API returns (obs, info)

for _ in range(500):
    env.render()
    action = env.action_space.sample()  # Random action: 0 (left) or 1 (right)
    ## action = 0 #always to left
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.02)  # Slow down so you can see
    if terminated or truncated:
        obs = env.reset()[0]
        print(f"terminated at {_}")

env.close()
