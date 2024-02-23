import gymnasium as gym
from tinygrad import Tensor, nn
from tinygrad.dtype import dtypes

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

if __name__ == "__main__":
    for i in range(EPISODES):
        actions, states, rewards, losses = [],[],[],[]
        while True:
            ep_len = 0
            reward = 0
            done = False
            obs, _ = env.reset()
            while not done: 
                ep_len += 1
                states.append(obs)
                Tensor.no_grad=True
                action = model(Tensor(obs)).multinomial().realize().item()
                Tensor.no_grad=False
                obs, rew, done, _, _= env.step(action)
                actions.append(action)
                reward += float(rew)
            rewards += [reward] * ep_len
            if len(actions) > BATCH_SIZE:
                break
        losses.append(trainStep(model, Tensor(states), Tensor(rewards), Tensor(actions, dtype=dtypes.int8)))
        print(f"episode {i} rewards = {sum(rewards)/len(rewards)}, avg_loss = {(sum(losses)/len(losses))}")


