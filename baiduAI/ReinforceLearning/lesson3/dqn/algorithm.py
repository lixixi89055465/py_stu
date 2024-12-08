import copy
import paddle.fluid as fluid
import parl
from parl import layers


class DQN(parl.Algorithm):
    def __init__(self, model, act_dim=None, gamma=None, lr=None):
        '''
        DQN algorithm
        Args :
        model (parl.Model): 定义Q函数的前向网络结构
        act_dim(int):action 空间的维度，即有几个action
        gamma(float): reward 的衰减因子
        lr(float):learning_rate ,学习率
        '''
        self.model = model
        self.target_model = copy.deepcopy(model)
        assert isinstance(act_dim, int)
        assert isinstance(gamma, float)
        assert isinstance(lr, float)
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr

    def predict(self, obs):
        """
        使用 self.model 的value 网络获取[Q(s,a1),Q(s,a2),...]
        """
        return self.model.value(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        '''
        使用DNA 算法更新self.model 的value 网络
        '''
        # 从target_model 中获取 max Q '的值，用于计算target_Q
        next_pred_value = self.target_model.value(next_obs)
        best_v = layers.reduce_max(next_pred_value, dim=1)
        best_v.stop_gradient = True  # 阻止梯度传递
        terminal = layers.cast(terminal, dtype='float32')
        target = reward + (1.0 - terminal) * self.gamma * best_v

        pred_value = self.model.value(obs)  # 获取Q预测值
        # 将action 转onehot 向量，比如:3=>[0,0,0,1,0]
        action_onehot = layers.one_hot(action, self.act_dim)
        action_onehot = layers.cast(action_onehot, dtype='float32')
        # 下面一行是逐元素相乘，拿到 action 对应的Q(s,a)
        # 比如：pred_value=[[2.3,5.7,1.2,3.9,1.4]],action_onehot=[[0,0,0,1,0]]
        # ==> pred_action_value=[[3.9]]
        pred_action_value = layers.reduce_sum(
            layers.elementwise_mul(action_onehot, pred_value), dim=1)

        # 计算Q(s,a)与target_Q 的均方差，得到loss
        cost = layers.square_error_cost(pred_action_value, target)
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.Adam(learning_rate=self.lr)  # 使用Adam 优化器
        optimizer.minimize(cost)
        return cost

    def sync_target(self):
        ''' 把self.model 的模型参数值同步到self.target_model
        '''
        self.model.sync_weights_to(self.target_model)

