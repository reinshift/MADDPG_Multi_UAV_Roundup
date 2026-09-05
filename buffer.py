# 多智能体经验回放：同步保存局部观测、联合状态与各智能体动作。
# 本文件注释描述当前实现；参数限制与特殊行为以具体代码为准。

import numpy as np

# 按相同时间索引保存所有智能体的一次联合转移，容量满后循环覆盖旧数据。
class MultiAgentReplayBuffer:
    # max_size 为容量，batch_size 为采样量；critic_dims 为拼接状态长度。
    # actor_dims 为各智能体观测长度列表；n_agents/n_actions 决定奖励与动作数组形状。
    def __init__(self, max_size, critic_dims, actor_dims, 
            n_actions, n_agents, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions

        # 共享数组形状分别为 [容量, 联合状态维度] 和 [容量, 智能体数]。
        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

        self.init_actor_memory()

    # 分别分配每个智能体的观测、下一观测和动作数组，允许观测维度不同。
    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_memory.append(
                            np.zeros((self.mem_size, self.n_actions)))


    # 写入一步经验：raw_obs 为局部观测列表，state 为其拼接后的联合状态。
    # action/reward/done 按智能体排列；带下划线的参数表示下一时刻数据。
    def store_transition(self, raw_obs, state, action, reward, 
                               raw_obs_, state_, done):
        # this introduces a bug: if we fill up the memory capacity and then
        # zero out our actor memory, the critic will still have memories to access
        # while the actor will have nothing but zeros to sample. Obviously
        # not what we intend.
        # In reality, there's no problem with just using the same index
        # for both the actor and critic states. I'm not sure why I thought
        # this was necessary in the first place. Sorry for the confusion!

        #if self.mem_cntr % self.mem_size == 0 and self.mem_cntr > 0:
        #    self.init_actor_memory()
        
        # Reuse one ring-buffer slot for every agent and the shared critic state.
        # 用取模得到循环写入位置；所有智能体和 Critic 数据必须使用同一槽位。
        index = self.mem_cntr % self.mem_size

        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    # 无放回采样 batch_size 条联合经验，调用前应通过 ready 检查。
    # 返回局部观测、联合状态、动作、奖励、下一局部观测、下一联合状态和终止标记。
    # 局部数据按智能体分组，各组首维为 batch_size；共享数据首维同样为 batch_size。
    def sample_buffer(self):
        # Only sample populated slots while the buffer is still filling.
        max_mem = min(self.mem_cntr, self.mem_size)

        # Shared indices keep all agents' observations and actions time-aligned.
        # 同一组随机下标应用于所有数组，保证一个样本中的联合信息来自同一时刻。
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        actor_states = []
        actor_new_states = []
        actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])

        return actor_states, states, actions, rewards, \
               actor_new_states, states_, terminal

    # 累计写入量达到 batch_size 时返回 True，否则隐式返回 None。
    # 调用者需保证 batch_size 不超过容量，才足以进行无放回采样。
    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True
