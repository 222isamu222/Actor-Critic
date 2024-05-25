import numpy as np
import gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L
import matplotlib.pyplot as plt

class Policy(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(action_size)
        
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))
        return x
    
class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 2
        
        self.memory = []
        self.pi = Policy(self.action_size)
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.pi)
        
    def get_action(self, state):
        state = state[np.newaxis, :]
        probs = self.pi(state)
        probs = probs[0]
        action = np.random.choice(len(probs), p=probs.data)
        return action, probs[action]
    
    def add(self, reward, prob):
        data = (reward, prob)
        self.memory.append(data)
    
    def update(self):
        self.pi.cleargrads()
        
        G, loss = 0, 0
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma * G
            loss += -F.log(prob) * G
        
        loss.backward()
        self.optimizer.update()
        self.memory = []

episodes = 3000
env = gym.make('CartPole-v0')
agent = Agent()
reward_history = []

for episode in range(episodes):
    state = env.reset(seed = 0)
    terminate, truncated = False, False
    sum_reward = 0
    
    while not terminate and not truncated:
        action, prob = agent.get_action(state if type(state[1]) != dict else state[0])
        next_state, reward, terminate, truncated, info = env.step(action)
        
        agent.add(action, prob)
        state = next_state
        sum_reward += reward
    
    agent.update()
    
    reward_history.append(sum_reward)
    if episode % 100 == 0:
        print("episode : {}, total reward : {:.1f}".format(episode, sum_reward))
        
plt.xlabel("Episode")
plt.ylabel('Total Reward')
plt.plot(range(len(reward_history)), reward_history)
plt.show()


env = gym.make('CartPole-v0', render_mode="human")
state = env.reset(seed=0)
terminate, truncated = False, False

while not terminate and not truncated:
    action, prob = agent.get_action(state if type(state[1]) != dict else state[0])
    next_state, reward, terminate, truncated, info = env.step(action)
    
    agent.add(action, prob)
    state = next_state