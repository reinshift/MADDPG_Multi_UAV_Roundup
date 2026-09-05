# 单个智能体的策略、价值网络、探索动作及目标网络同步。
# 本文件注释描述当前实现；参数限制与特殊行为以具体代码为准。

import torch as T
from networks import ActorNetwork, CriticNetwork
import numpy as np

# 管理一名智能体的四个网络；Actor 使用局部观测，Critic 使用联合信息。
class Agent:
    # actor_dims/critic_dims 为局部/联合输入维度，n_actions 为每个动作维度。
    # alpha/beta 为策略/价值学习率，fc1/fc2 为隐藏层宽度；gamma 为折扣，tau 为软更新比例。
    # agent_idx 区分权重文件，chkpt_dir 指定保存目录，n_agents 决定联合动作宽度。
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
                    alpha=0.0001, beta=0.0002, fc1=128,
                    fc2=128, gamma=0.99, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, 
                                  chkpt_dir=chkpt_dir,  name=self.agent_name+'_actor')
        self.critic = CriticNetwork(beta, critic_dims, 
                            fc1, fc2, n_agents, n_actions, 
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_critic')
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                        chkpt_dir=chkpt_dir, 
                                        name=self.agent_name+'_target_actor')
        self.target_critic = CriticNetwork(beta, critic_dims, 
                                            fc1, fc2, n_agents, n_actions,
                                            chkpt_dir=chkpt_dir,
                                            name=self.agent_name+'_target_critic')

        # 初次同步使用完整复制，确保目标网络与在线网络从相同参数开始。
        self.update_network_parameters(tau=1)

    # 将 observation 转成单样本张量；time_step 控制探索噪声衰减。
    # evaluate=True 关闭噪声；返回长度 n_actions 的 NumPy 向量，范数不超过 0.04。
    def choose_action(self, observation, time_step, evaluate=False):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        actions = self.actor.forward(state)

        # exploration
        max_noise = 0.75
        min_noise = 0.01
        decay_rate = 0.999995

        # 噪声幅度随全局步数指数衰减，下限为 0.01；各动作维度独立均匀采样。
        noise_scale = max(min_noise, max_noise * (decay_rate ** time_step))
        noise = 2 * T.rand(self.n_actions).to(self.actor.device) - 1 # [-1,1)
        if not evaluate:
            noise = noise_scale * noise
        else:
            noise = 0 * noise
        
        action = actions + noise
        action_np = action.detach().cpu().numpy()[0]
        # 对整个动作向量做范数裁剪，保留方向；评估时也应用此限制。
        magnitude = np.linalg.norm(action_np)
        if magnitude > 0.04:
            action_np = action_np / magnitude * 0.04
        return action_np

    # 按 target = tau * online + (1 - tau) * target 同步两组目标网络。
    # tau 未指定时采用实例配置；tau=1 表示完整复制在线参数。
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    # 保存在线和目标 Actor/Critic 的四份权重。
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    # 从各网络预设路径恢复四份权重；文件不存在时由加载器抛出异常。
    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
