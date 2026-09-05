# MADDPG 协调器：分散选择动作，使用联合状态和联合动作集中训练。
# 本文件注释描述当前实现；参数限制与特殊行为以具体代码为准。

import os
import torch as T
import torch.nn.functional as F
from agent import Agent
# from torch.utils.tensorboard import SummaryWriter

# 为每个智能体维护独立网络，通过同一批联合经验执行 MADDPG 更新。
class MADDPG:
    # actor_dims 为各局部观测长度，critic_dims 为联合状态长度；scenario 区分模型目录。
    # alpha/beta 传给各 Agent；当前构造调用没有转发 fc1/fc2/gamma/tau，实际使用 Agent 默认值。
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, 
                 scenario='simple',  alpha=0.01, beta=0.02, fc1=128, 
                 fc2=128, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario
        # self.writer = SummaryWriter(log_dir=os.path.join(chkpt_dir, 'logs'))

        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,  
                            n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                            chkpt_dir=chkpt_dir))

    # 逐一创建模型目录并保存所有智能体的在线网络及目标网络。
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            os.makedirs(os.path.dirname(agent.actor.chkpt_file), exist_ok=True)
            agent.save_models()

    # 逐一加载全部智能体模型，目录由构造时的 chkpt_dir 和 scenario 决定。
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    # 按 raw_obs 的智能体顺序生成动作列表，将步数和评估标记传给各策略。
    def choose_action(self, raw_obs, time_step, evaluate):# timestep for exploration
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx],time_step, evaluate)
            actions.append(action)
        return actions

    # 从 memory 抽取共享批次，依次更新每个 Critic 和 Actor，最后软更新目标网络。
    # total_steps 目前只在注释掉的 TensorBoard 日志示例中使用。
    def learn(self, memory, total_steps):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device

        # 联合状态为 [批量, critic_dims]；动作张量为 [智能体数, 批量, n_actions]。
        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        old_agents_actions = []
    
        for agent_idx, agent in enumerate(self.agents):

            new_states = T.tensor(actor_new_states[agent_idx], 
                                dtype=T.float).to(device)

            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)
            old_agents_actions.append(actions[agent_idx])

        # Centralized critics receive the joint action in agent-index order.
        # 拼接后联合动作形状为 [批量, 智能体数*n_actions]，列顺序与智能体索引一致。
        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)

        for agent_idx, agent in enumerate(self.agents):
            # 目标 Q 值只用于监督 Critic，不对目标网络反向传播。
            with T.no_grad():
                critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
                # Bellman 目标为即时奖励加折扣后的下一状态价值；终止时取消后半项。
                # 当前用 dones 的第 0 列作为共同终止标记，依赖所有智能体同步结束。
                target = rewards[:,agent_idx] + (1-dones[:,0].int())*agent.gamma*critic_value_

            critic_value = agent.critic.forward(states, old_actions).flatten()
            
            # 最小化预测 Q 与 Bellman 目标的均方误差。
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()
            agent.critic.scheduler.step()

            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)
            # Replace only this agent's replay action to optimize its policy.
            oa = old_actions.clone()
            oa[:,agent_idx*self.n_actions:agent_idx*self.n_actions+self.n_actions] = agent.actor.forward(mu_states)            
            # 最小化负 Q 等价于提高当前策略的价值；其他智能体动作保持为回放动作。
            actor_loss = -T.mean(agent.critic.forward(states, oa).flatten())
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()
            agent.actor.scheduler.step()

            # self.writer.add_scalar(f'Agent_{agent_idx}/Actor_Loss', actor_loss.item(), total_steps)
            # self.writer.add_scalar(f'Agent_{agent_idx}/Critic_Loss', critic_loss.item(), total_steps)

            # for name, param in agent.actor.named_parameters():
            #     if param.grad is not None:
            #         self.writer.add_histogram(f'Agent_{agent_idx}/Actor_Gradients/{name}', param.grad, total_steps)
            # for name, param in agent.critic.named_parameters():
            #     if param.grad is not None:
            #         self.writer.add_histogram(f'Agent_{agent_idx}/Critic_Gradients/{name}', param.grad, total_steps)
            
        for agent in self.agents:    
            # 完成本轮在线更新后缓慢跟踪参数，使下一轮训练目标更平稳。
            agent.update_network_parameters()
