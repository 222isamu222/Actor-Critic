import numpy as np
import gym
from dezero import Model
import dezero.functions as F
import dezero.layers as L
from dezero import optimizers
import matplotlib.pyplot as plt

#方策を基にしたニューラルネットワーク <- 方策勾配法
class PolicyNet(Model):
    def __init__(self, action_size=2):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(action_size)
        
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)	#←? => 線形変換
        x = F.softmax(x)
        return x	#ソフトマックス関数なので確率が出力
    
#価値関数を基にしたニューラルネットワーク <- Q学習
class ValueNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(1)
        
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x	#Q関数が出力

class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.0005
        self.action_size = 2
        
        self.pi = PolicyNet()
        self.v = ValueNet()
        self.optimizer_pi = optimizers.Adam(self.lr_pi).setup(self.pi)
        self.optimizer_v = optimizers.Adam(self.lr_v).setup(self.v)
        
    def get_action(self, state):
        state = state[np.newaxis, :]	#バッチ軸の追加
        probs = self.pi(state)
        probs = probs[0]
        action = np.random.choice(len(probs), p=probs.data)
        return action, probs[action]
    
    def update (self, state, action_prob, reward, next_state, terminate, truncated):
        #バッチ軸の追加
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]
        
        #self.vの損失
        target = reward + self.gamma * self.v(next_state) * (1 - (terminate | truncated))
        target.unchain()
        v = self.v(state)
        loss_v = F.mean_squared_error(target, v)
        
        #self.piの損失
        delta = target - v
        delta.unchain()
        loss_pi = -F.log(action_prob) * delta
        
        self.v.cleargrads()
        self.pi.cleargrads()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.update()
        self.optimizer_pi.update()
        
episodes = 3000
env = gym.make('CartPole-v0')
agent = Agent()
reward_history = []

for episode in range(episodes):
    state = env.reset(seed=0)
    terminate, truncated = False, False
    total_reward = 0
    
    while not terminate and not truncated:
        action, prob = agent.get_action(state if type(state[1]) != dict else state[0])
        next_state, reward, terminate, truncated, info = env.step(action)
        
        agent.update(state if type(state[1]) != dict else state[0], prob, reward, next_state, terminate, truncated)
        state = next_state
        total_reward += reward
        
    reward_history.append(total_reward)
    if episode % 100 == 0:
        print("episode : {}, total reward : {:.1f}".format(episode, total_reward))
    
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.plot(range(len(reward_history)), reward_history)
plt.show()


env = gym.make('CartPole-v0', render_mode="human")
state = env.reset(seed=0)
terminate, truncated = False, False

while not terminate and not truncated:
    action, prob = agent.get_action(state if type(state[1]) != dict else state[0])
    next_state, reward, terminate, truncated, info = env.step(action)
    
    agent.update(state if type(state[1]) != dict else state[0] , prob, reward, next_state, terminate, truncated)
    state = next_state
