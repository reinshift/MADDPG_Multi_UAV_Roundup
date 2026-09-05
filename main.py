# 训练与评估入口；当前 evaluate=True，默认读取已有模型进行评估。
# 本文件注释描述当前实现；参数限制与特殊行为以具体代码为准。

import numpy as np
from maddpg import MADDPG
from sim_env import UAVEnv
from buffer import MultiAgentReplayBuffer
import time
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
from PIL import Image
warnings.filterwarnings('ignore')

# 按智能体顺序展平并拼接局部观测，构成 Critic 使用的一维联合状态。
def obs_list_to_state_vector(obs):
    state = np.hstack([np.ravel(o) for o in obs])
    return state

# 将环境返回的 RGBA 像素数组转换为 RGB 后写入 filename；父目录由调用方创建。
def save_image(env_render, filename):
    # Convert the RGBA buffer to an RGB image
    image = Image.fromarray(env_render, 'RGBA')  # Use 'RGBA' mode since the buffer includes transparency
    image = image.convert('RGB')  # Convert to 'RGB' if you don't need transparency
    
    image.save(filename)

if __name__ == '__main__':

    env = UAVEnv()
    # print(env.info)
    n_agents = env.num_agents
    actor_dims = []
    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    # 联合状态包含全部局部观测，默认长度为 26*3+23=101。
    critic_dims = sum(actor_dims)

    # action space is a list of arrays, assume each agent has same action space
    n_actions = 2
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=128, fc2=128,
                           alpha=0.0001, beta=0.003, scenario='UAV_Round_up',
                           chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
                        n_actions, n_agents, batch_size=256)

    PRINT_INTERVAL = 100
    N_GAMES = 5000
    MAX_STEPS = 100
    total_steps = 0
    score_history = []
    target_score_history = []
    # 训练时设为 False；评估模式加载权重、禁用探索并跳过网络学习。
    evaluate = True
    best_score = -30

    if evaluate:
        maddpg_agents.load_checkpoint()
        print('----  evaluating  ----')
    else:
        print('----training start----')
    
    for i in range(N_GAMES):
        obs = env.reset()
        score = 0
        score_target = 0
        dones = [False]*n_agents
        episode_step = 0
        while not any(dones):
            if evaluate:
                # env.render()
                env_render = env.render()
                if episode_step % 10 == 0:
                    # Save the image every 10 episode steps
                    filename = f'images/episode_{i}_step_{episode_step}.png'
                    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Create directory if it doesn't exist
                    save_image(env_render, filename)
                # time.sleep(0.01)
            actions = maddpg_agents.choose_action(obs,total_steps,evaluate)
            obs_, rewards, dones = env.step(actions)

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            # 计数从零开始且在步末增加，因此此条件允许执行编号 0 到 MAX_STEPS 的步骤。
            if episode_step >= MAX_STEPS:
                dones = [True]*n_agents

            # 当前实现评估时也写入回放缓存；训练每 10 个全局步尝试更新一次。
            memory.store_transition(obs, state, actions, rewards, obs_, state_, dones)

            if total_steps % 10 == 0 and not evaluate:
                maddpg_agents.learn(memory,total_steps)

            obs = obs_
            # 这里的 0:2 只累计索引 0、1 的奖励，不包含第三架追捕无人机。
            score += sum(rewards[0:2])
            score_target += rewards[-1]
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        target_score_history.append(score_target)
        # 最近最多 100 个回合的平均分用于进度输出和训练模型择优保存。
        avg_score = np.mean(score_history[-100:])
        avg_target_score = np.mean(target_score_history[-100:])
        if not evaluate:
            if i % PRINT_INTERVAL == 0 and i > 0 and avg_score > best_score:
                print('New best score',avg_score ,'>', best_score, 'saving models...')
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score),'; average target score {:.1f}'.format(avg_target_score))
    
    # save data
    # 每次运行将整条回合分数序列作为一行保存；文件已存在时追加。
    file_name = 'score_history.csv'
    if not os.path.exists(file_name):
        pd.DataFrame([score_history]).to_csv(file_name, header=False, index=False)
    else:
        with open(file_name, 'a') as f:
            pd.DataFrame([score_history]).to_csv(f, header=False, index=False)