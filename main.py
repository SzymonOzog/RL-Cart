import gymnasium as gym
from tinygrad import Tensor, nn, TinyJit
from tinygrad.dtype import dtypes
import numpy as np

EPISODES = 50
BATCH_SIZE = 5000
LR = 1e-2
GAMMA = 0.99

class RLModel:
    def __init__(self):
        self.fc1 = nn.Linear(in_features=4, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=2)

    def __call__(self, x):
        x = self.fc1(x).tanh()
        x = self.fc2(x)
        return x.softmax()

def vanilla_loss(model, buffer):
    act = model(Tensor(buffer["states"]))
    taken = act[Tensor.arange(act.shape[0]),Tensor(buffer["actions"], dtype=dtypes.uint8)]
    loss = -(taken.log() * Tensor.stack(buffer["rewards"])).mean()
    return loss

def actor_critic_loss(model, buffer):
    act = model(Tensor(buffer["states"]))
    values = critic(Tensor(buffer["states"]))
    advantage = Tensor(buffer["rewards"], dtype=dtypes.float) - values
    taken = act[Tensor.arange(act.shape[0]),Tensor(buffer["actions"], dtype=dtypes.uint8)]
    loss = -(taken.log() * advantage).mean()
    critic_loss = advantage.pow(2).mean()
    return loss + critic_loss

class Critic:
    def __init__(self):
        self.fc1 = nn.Linear(in_features=4, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=1)

    def __call__(self, x):
        x = self.fc1(x).tanh()
        x = self.fc2(x)
        return x.flatten()

def cummulative_rewards(ep_rews):
    return [sum(ep_rews)] * len(ep_rews)

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

def rewards_to_go(ep_rews, gamma=0.99):
    ep_rews.reverse()
    return (np.flip(np.cumsum(ep_rews)) * np.power(gamma,np.arange(len(ep_rews)))).tolist()

env = gym.make("CartPole-v1")
model = RLModel()
critic = Critic()
opt = nn.optim.Adam(nn.state.get_parameters(model), LR)

buffer = {}

@TinyJit
def get_action(obs:Tensor) -> Tensor:
    Tensor.no_grad=True
    a = model(obs).multinomial().realize()
    Tensor.no_grad=False
    return a

@TinyJit
def train_step():
    with Tensor.train():
        opt.zero_grad()
        loss = actor_critic_loss(model, buffer)
        loss.backward()
        opt.step()
    return loss

if __name__ == "__main__":
    for i in range(EPISODES):
        get_action.reset()
        train_step.reset()
        buffer["actions"] = []
        buffer["states"] = []
        buffer["losses"] = []
        buffer["rewards"] = []
        buffer["lengths"] = []
        while True:
            ep_rews = []
            done = False
            obs, _ = env.reset()
            while not done: 
                buffer["states"].append(obs)
                action = get_action(Tensor(obs)).item()
                obs, rew, done, _, _= env.step(action)
                buffer["actions"].append(action)
                ep_rews.append(rew)
                buffer["lengths"].append(len(ep_rews))
            buffer["rewards"] += (rewards_to_go(ep_rews, gamma=GAMMA))
            if len(buffer["actions"]) > BATCH_SIZE:
                break
        buffer["losses"].append(train_step().numpy())
        print(f"episode {i}, avg_loss = {(sum(buffer['losses'])/len(buffer['losses']))}, longest episode = {max(buffer['lengths'])}")

