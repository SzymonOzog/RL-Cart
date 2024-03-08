import gymnasium as gym
from tinygrad import Tensor, nn, TinyJit
from tinygrad.dtype import dtypes
import numpy as np
import matplotlib.pyplot as plt

EPISODES = 500
BATCH_SIZE = 512
LR = 1e-2
TRAIN_STEPS = 10
GAMMA = 0.99
EPSILON = 0.2
ENTROPY_COEF = 0.0005
SOLVED_THRESHOLD = 10000

class RLModel:
    def __init__(self):
        self.fc1 = nn.Linear(in_features=4, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=2)

    def __call__(self, x):
        x = self.fc1(x).tanh()
        x = self.fc2(x)
        return x.softmax()

def vanilla_loss(model, buffer):
    act = model(buffer["states"])
    taken = act[Tensor.arange(act.shape[0]),buffer["actions"]]
    loss = -(taken.log() * buffer["rewards"]).mean()
    return loss

def actor_critic_loss(model, buffer):
    act = model(buffer["states"])
    values = critic(buffer["states"])
    advantage = buffer["rewards"] - values
    taken = act[Tensor.arange(act.shape[0]),buffer["actions"]]
    loss = -(taken.log() * advantage).mean()
    critic_loss = advantage.pow(2).mean()
    return loss + critic_loss

def ppo_loss(model, buffer):
    samples = Tensor.randint(BATCH_SIZE, high=buffer["rewards"].shape[0]).realize()
    act = model(buffer["states"][samples])
    taken = act[Tensor.arange(act.shape[0]),buffer["actions"][samples]]
    
    values = critic(buffer["states"][samples])
    advantage = buffer["rewards"][samples] - values

    ratio = taken / buffer["old_probs"][samples]
    clip = ratio.clip(1 - EPSILON, 1 + EPSILON)
    loss = -Tensor.minimum(ratio * advantage, clip * advantage).mean()

    entropy_loss = (act * act.log()).mean()

    critic_loss = advantage.pow(2).mean()
    return loss + critic_loss + ENTROPY_COEF * entropy_loss

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

def rewards_to_go(ep_rews, gamma=0.99, standardize=False):
    ep_rews.reverse()
    rews = (np.flip(np.cumsum(ep_rews)) * np.power(gamma,np.arange(len(ep_rews))))
    if standardize:
        rews = (rews - np.mean(rews)) / (np.std(rews) + 1e-8)
    return rews.tolist()

@TinyJit
def get_action(obs:Tensor) -> Tensor:
    Tensor.no_grad=True
    a = model(obs).multinomial().realize()
    Tensor.no_grad=False
    return a

@TinyJit
def train_step() -> Tensor:
    with Tensor.train():
        opt.zero_grad()
        loss = loss_fn(model, buffer)
        loss.backward()
        opt.step()
    return loss


def show_results():
    env = gym.make("CartPole-v1", render_mode="human")
    obs, _ = env.reset()
    while True:
        env.render()
        action = get_action(Tensor(obs)).item()
        obs, _, done, _, _= env.step(action)
        if done:
            break
    env.close()

env = gym.make("CartPole-v1")
model = RLModel()
critic = Critic()
opt = nn.optim.Adam(nn.state.get_parameters(model), LR)
loss_fn = ppo_loss
buffer = {}

def run_training():
    step = 0
    steps, losses, lengths = [], [], []
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
                step += 1
                buffer["states"].append(obs)
                action = get_action(Tensor(obs)).item()
                obs, rew, done, _, _= env.step(action)
                buffer["actions"].append(action)
                ep_rews.append(rew)
                if len(ep_rews) > SOLVED_THRESHOLD:
                    return True, steps, losses, lengths

            buffer["lengths"].append(len(ep_rews))

            buffer["rewards"] += (rewards_to_go(ep_rews, gamma=GAMMA, standardize=True))
            if len(buffer["actions"]) > BATCH_SIZE:
                break
        buffer["actions"] = Tensor(buffer["actions"], dtype=dtypes.uint8)
        buffer["states"] = Tensor(buffer["states"])
        buffer["rewards"] = Tensor(buffer["rewards"])
        buffer["lengths"] = Tensor(buffer["lengths"])
        buffer["old_probs"] = model(buffer["states"])[Tensor.arange(buffer["states"].shape[0]),buffer["actions"]]
        for _ in range(TRAIN_STEPS):
            buffer["losses"].append(train_step().numpy())
        # print(f"episode {i}, avg_loss = {(sum(buffer['losses'])/len(buffer['losses']))}, longest episode = {buffer['lengths'].max().numpy()}")
        steps.append(step)
        losses.append(sum(buffer["losses"]) / len(buffer["losses"]))
        lengths.append(buffer["lengths"].max().numpy())
    return False, steps, losses, lengths

if __name__ == "__main__":
    outcomes = {}
    for current_loss in ["vanilla policy gradients", "actor critic", "ppo"]:
        outcomes[current_loss] = []
    for lr in [1e-1, 1e-2, 1e-3, 1e-4]:
        for gamma in [0.99, 0.95, 0.9, 0.85]:
            solved, steps, losses, lengths = [], [], [], []
            for current_loss in ["vanilla policy gradients", "actor critic", "ppo"]:
                exploded = 0
                loss_fn = vanilla_loss if current_loss == "vanilla policy gradients" else actor_critic_loss if current_loss == "actor critic" else ppo_loss
                TRAIN_STEPS = 10 if current_loss == "ppo" else 1
                LR = lr
                GAMMA = gamma
                model = RLModel()
                critic = Critic()
                opt = nn.optim.Adam(nn.state.get_parameters(model), LR)
                try:
                    so, st, lo, le = run_training()
                    plt.plot(st, le)
                    print(f"loss = {current_loss}, lr = {lr}, gamma = {gamma}, solved = {so}, steps = {st[-1]}, length = {le[-1]}")
                    outcomes[current_loss].append(so)
                except Exception as e:
                    print(f"exploded with {e}")
                    exploded += 1
                print(f"loss = {current_loss}, solved = {sum(solved)}/{len(solved)}, exploded = {exploded}")
            plt.title(f"gamma = {gamma}, lr = {lr}")
            plt.xlabel("steps")
            plt.ylabel("episode length")
            plt.legend(["vanilla policy gradients", "actor critic", "ppo"])
            plt.savefig(f"results_{gamma}_{lr}.png")
            plt.clf()
    for loss, results in outcomes.items():
        print(f"{loss} solved {sum(results)}/{len(results)}")
