# Actor 与 Critic 的全连接网络、优化器、学习率调度和权重读写。
# 本文件注释描述当前实现；参数限制与特殊行为以具体代码为准。

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 集中式价值网络：输入联合状态与所有智能体动作，输出一个 Q 值。
class CriticNetwork(nn.Module):
    # beta 为学习率；input_dims 为联合状态维度，联合动作宽度为 n_agents*n_actions。
    # fc1_dims/fc2_dims 指定隐藏层宽度，chkpt_dir/name 组成权重文件路径。
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, 
                    n_agents, n_actions, name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims+n_agents*n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        # 每调用 scheduler.step() 5000 次，将 Critic 学习率乘以 0.33。
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.33)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    # state 和 action 均以批次为首维；拼接特征后返回形状 [batch, 1] 的 Q 值。
    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    # 创建父目录并保存 state_dict；这里不保存优化器或调度器状态。
    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)    
        T.save(self.state_dict(), self.chkpt_file)

    # 加载网络权重；当前调用没有指定 map_location，设备映射采用 PyTorch 默认行为。
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


# 分散式策略网络：将单个智能体的局部观测映射到连续动作。
class ActorNetwork(nn.Module):
    # alpha 为学习率，input_dims 为局部观测长度，n_actions 为输出动作长度。
    # fc1_dims/fc2_dims 指定隐藏层宽度，chkpt_dir/name 组成权重文件路径。
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, 
                 n_actions, name, chkpt_dir):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        # 每调用 scheduler.step() 1000 次，将 Actor 学习率乘以 0.8。
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.8)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    # 输入 [batch, input_dims]，输出 [batch, n_actions]；Softsign 将每一维压到 (-1, 1)。
    # 实际执行动作的范数限制在 Agent.choose_action 中处理。
    def forward(self, state):
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        pi = nn.Softsign()(self.pi(x)) # [-1,1]

        return pi

    # 创建父目录并保存 state_dict；这里不保存优化器或调度器状态。
    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        T.save(self.state_dict(), self.chkpt_file)

    # 加载网络权重；当前调用没有指定 map_location，设备映射采用 PyTorch 默认行为。
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))

