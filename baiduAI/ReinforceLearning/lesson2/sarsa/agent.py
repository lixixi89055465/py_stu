import numpy as np


class SarsaAgent(object):
    def __init__(self,
                 obs_n,
                 act_n,
                 learning_rate=0.01,
                 gamma=0.9,
                 e_greed=0.1):
        self.act_n = act_n  # 动作维度
        self.lr = learning_rate  # 学习率
        self.gamma = gamma  # reward 的衰减率
        self.epsilon = e_greed  # 按一定的概率随机选动作
        self.Q = np.zeros((obs_n, act_n))

    # 根据输入观察值，采样输出的动作值，带探索
    def sample(self, obs):
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):  # 根据table的Q值选动作
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)  # 有一定概率随机探索选取一个动作
        return action

    # 根据输入观察值，预测输出的动作
    def predict(self, obs):
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]  # maxQ 可能对应多个action
        action = np.random.choice(action_list)
        return action

    def learn(self, obs, action, reward, next_obs, next_action, done):
        '''
        on-policy
        obs:  交互前的obs,s_t
        action:  本次交互选择的action,a_t
        reward:  本次动作获得的奖励
        next_obs: 本次交互后的obs,s_t+1
        next_action: 根据当前Q表格，针对next_obs 会选择的动作a_t+1
        done:  episode 是否结束
        '''
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward  # 没有下一个状态了
        else:
            target_Q = reward + self.gamma * self.Q[next_obs, next_action]  # Sarsa
        self.Q[obs, action] += self.lr * (target_Q - predict_Q)  # 修正Q

    def save(self):
        npy_file = './Q_tabel.npy'
        np.save(npy_file, self.Q)
        print(npy_file + '  saved')

    def restore(self, npy_file='./q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded')
