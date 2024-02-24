import gymnasium as gym
from tinygrad import Tensor, nn
from tinygrad.dtype import dtypes
import numpy as np

EPISODES = 50
BATCH_SIZE = 5000
LR = 1e-2

class RLModel:
    def __init__(self):
        self.fc1 = nn.Linear(in_features=4, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=2)

    def __call__(self, x):
        x = self.fc1(x).tanh()
        x = self.fc2(x)
        return x.softmax()

def trainStep(model, state, reward, actions):
    with Tensor.train():
        act = model(state)
        taken = act[Tensor.arange(act.shape[0]),actions]
        loss = -(taken.log()*(reward)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.numpy()

env = gym.make("CartPole-v1")
model = RLModel()
opt = nn.optim.Adam(nn.state.get_parameters(model), LR)

def cummulative_rewards(ep_rews):
    return [sum(ep_rews)] * len(ep_rews)

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

def rewards_to_go(ep_rews):
    ep_rews.reverse()
    return np.flip(np.cumsum(ep_rews)).tolist()

if __name__ == "__main__":
    for i in range(EPISODES):
        actions, states, rewards, losses, lengths = [],[],[],[],[]
        while True:
            ep_rews = []
            done = False
            obs, _ = env.reset()
            while not done: 
                states.append(obs)
                Tensor.no_grad=True
                action = model(Tensor(obs)).multinomial().realize().item()
                Tensor.no_grad=False
                obs, rew, done, _, _= env.step(action)
                actions.append(action)
                ep_rews.append(rew)
                lengths.append(len(ep_rews))
            rewards += rewards_to_go(ep_rews)
            if len(actions) > BATCH_SIZE:
                break
        losses.append(trainStep(model, Tensor(states), Tensor(rewards), Tensor(actions, dtype=dtypes.int8)))
        print(f"episode {i} rewards = {sum(rewards)/len(rewards)}, avg_loss = {(sum(losses)/len(losses))}, longest episode = {max(lengths)}")
