import gym
from gridworld import CliffWalkingWapper, FrozenLakeWapper
from agent import SarsaAgent
import time


def run_episode(env, agent, render=False):
    total_steps = 0  # 记录每个episode 走了多少step
    total_reward = 0
    obs = env.reset()  # 重制环境，重新开一局(即开始新的一个episode)
    action = agent.sample(obs)  # 根据算法选择一个动作
    while True:
        # time.sleep(0.5)
        next_obs, reward, done, _ = env.step(action)  # 与环境进行一个交互
        next_action = agent.sample(next_obs)  # 根据算法选择一个动作
        # 训练Sarsa 算法
        agent.learn(obs, action, reward, next_obs, next_action, done)
        action = next_action
        obs = next_obs  # 存储上一个观测值
        total_reward += reward
        total_steps += 1  # 计算step 数
        if render:
            env.render()  # 渲染新的一帧图片
        if done:
            break
    return total_reward, total_steps


def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
    while True:
        action = agent.predict(obs)  # greedy
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        obs = next_obs
        time.sleep(0.5)
        env.render()
        if done:
            print('test reward = %.1f' % (total_reward))
            break


def main():
    env = gym.make('FrozenLake-v0', is_slippery=True)  # 0 left,1 down,2 right, 3 up
    # env = FrozenLakeWapper(env)
    env = CliffWalkingWapper(env)
    agent = SarsaAgent(
        obs_n=env.observation_space.n,
        act_n=env.action_space.n,
        learning_rate=0.1,
        gamma=0.9,
        e_greed=0.1,
    )
    is_render = False
    for episode in range(5000):
        ep_reward, ep_steps = run_episode(env, agent, is_render)
        print('Episode %s : steps = %s, reward %.1f' % (episode, ep_steps, ep_reward))
        # 每隔20个episode 渲染一下看看看效果
        # time.sleep(0.5)
        if episode%20==0:
            is_render=True
        else:
            is_render=False
    # 训练结束，查看算法效果
    test_episode(env,agent)



if __name__ == "__main__":
    main()
