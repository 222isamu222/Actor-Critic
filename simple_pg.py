import numpy as np
import gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L

#方策(Policy)を基にニューラルネットワークを実装
class Policy(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(action_size)
        
    def forward(self, x):
        x = F.relu(self.l1(x))	#output = W*x + b を行ってreluで非線形データを活性化
        x = F.softmax(self.l2(x))	#最終出力 = ソフトマックス関数 よって，returnするのは各行動に対する「確率」
        return x
    

class Agent:
    def __init__(self):
        self.gamma =0.98
        self.lr = 0.0002
        self.action_size = 2
        
        self.memory = []
        self.pi = Policy(self.action_size)
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.pi)	#?
    
    #stateにおける行動を方策勾配法で求める．
    #self.pi(state)でニューラルネットワークの順伝搬してから，
    #ソフトマックス関数から確率分布probsを得て，その確率分布に従って行動を1つサンプリング
    def get_action(self, state):
        state = state[np.newaxis, :]	#バッチ処理用の軸を追加
        probs = self.pi(state)
        probs = probs[0]
        action = np.random.choice(len(probs), p=probs.data)
        return action, probs[action]
    
    #sampling：r～π_Θ
    def add(self, reward, prob):
        data = (reward, prob)
        self.memory.append(data)
    
    #ゴール到達時に呼ばれる更新メソッド
    #1.収益G(τ) = R_0 + γR_1 + γ^2R_2 + ... + γ^TR_T を求める + 報酬を逆向きにたどると効率がいい(ループ実行時のG + memory内の報酬でいいため)
    #2.損失関数lossの計算 = 各時刻において目的関数 J(Θ) = F.log(prob) にマイナスを掛けて-J(Θ)とし，そこに重みである収益Gを掛ける
    #目的関数を損失関数にする理由：損失関数の勾配降下法の最適化手法と．目的関数の方策上昇法を用いた最大値の探索のいいとこどりをするため
    def update(self):
        self.pi.cleargrads()
        
        G, loss = 0,0
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma * G
        
        for reward, prob in self.memory:
            loss += -F.log(prob) * G
        
        loss.backward()
        self.optimizer.update()
        self.memory = []	#メモリリセット
    
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
        next_state, reward, terminate, trucated, into = env.step(action)
        
        agent.add(reward, prob)
        state = next_state
        total_reward += reward
        
    agent.update()
    reward_history.append(total_reward)


env = gym.make('CartPole-v0',render_mode="human")
state = env.reset(seed=0)
terminate, truncated = False, False
total_reward = 0

while not terminate and not truncated:
	action, prob = agent.get_action(state if type(state[1]) != dict else state[0])
	next_state, reward, terminate, trucated, into = env.step(action)
	
	agent.add(reward, prob)
	state = next_state
	total_reward += reward
