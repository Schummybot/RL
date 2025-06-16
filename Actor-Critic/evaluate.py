import gym
import time
import torch
import torch.nn as nn
class ActorNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(ActorNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, x):
        out = self.net(x)
        return out
class CriticNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CriticNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        out = self.net(x)
        return out
# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(ActorCritic, self).__init__()
        # self.shared = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(),
        # )
        self.actor = ActorNet(input_dim, hidden_dim, action_dim)     # logits for actions
        self.critic = CriticNet(input_dim, hidden_dim)             # state value

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
env = gym.make("CartPole-v1", render_mode="human")  # 'human' to open a window
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
hidden_dim = 128
model = ActorCritic(obs_dim, hidden_dim, act_dim).to(device)
model = torch.load("cartpole.pt")
model.eval()
actor = model.actor
for _ in range(500):
    obs = env.reset()[0]
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
    logits = actor(obs_tensor)
    probs = torch.softmax(logits, dim=-1)
    action = torch.argmax(probs).item()
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.05)  # Slow down so you can see
    if terminated or truncated:
        obs = env.reset()[0]
        print(f"terminated at {_}")

env.close()