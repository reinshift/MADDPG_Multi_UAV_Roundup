# 二维无人机围捕环境：运动更新、局部观测、奖励、测距和渲染。
# 本文件注释描述当前实现；参数限制与特殊行为以具体代码为准。

import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.cm as cm
import matplotlib.image as mpimg
from gymnasium import spaces
from math_tool import *
import matplotlib.backends.backend_agg as agg
from PIL import Image
import random
import copy

# 默认三架追捕无人机加一个逃逸目标，最后一个索引始终代表目标。
# 观测空间、智能体名称及奖励几何固定为该配置，num_agents 并未实现任意数量泛化。
class UAVEnv:
    # length 为方形场地边长，num_obstacle 为圆障碍数量，num_agents 默认 4。
    # 设置运动参数、16 束传感器、空间描述和轨迹缓存；初始位置由 reset 生成。
    def __init__(self,length=2,num_obstacle=3,num_agents=4):
        self.length = length # length of boundary
        self.num_obstacle = num_obstacle # number of obstacles
        self.num_agents = num_agents
        self.time_step = 0.5 # update time step
        # 追捕者/目标使用不同速度上限；a_max 字段仅保存配置，step 未按它裁剪输入。
        self.v_max = 0.1
        self.v_max_e = 0.12
        self.a_max = 0.04
        self.a_max_e = 0.05
        self.L_sensor = 0.2
        self.num_lasers = 16 # num of laserbeams
        self.multi_current_lasers = [[self.L_sensor for _ in range(self.num_lasers)] for _ in range(self.num_agents)]
        self.agents = ['agent_0','agent_1','agent_2','target']
        self.info = np.random.get_state() # get seed
        self.obstacles = [obstacle() for _ in range(self.num_obstacle)]
        self.history_positions = [[] for _ in range(num_agents)]

        self.action_space = {
            'agent_0': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'agent_1': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'agent_2': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'target': spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
            } # action represents [a_x,a_y]
        # 这些 Box 描述固定观测维度；实际数据由 get_multi_obs 组装。
        self.observation_space = {
            'agent_0': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            'agent_1': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            'agent_2': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            'target': spaces.Box(low=-np.inf, high=np.inf, shape=(23,))
        }
        

    # 随机初始化追捕者位置，将目标放到 [0.5, 1.75]，清零速度和历史轨迹。
    # 刷新传感器并返回观测列表；不会重新生成已有障碍物。
    def reset(self):
        SEED = random.randint(1,1000)
        # 这里只设置 Python random；位置使用 NumPy 随机源，因此不固定其采样序列。
        random.seed(SEED)
        self.multi_current_pos = []
        self.multi_current_vel = []
        self.history_positions = [[] for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            if i != self.num_agents - 1: # if not target
                self.multi_current_pos.append(np.random.uniform(low=0.1,high=0.4,size=(2,)))
            else: # for target
                # self.multi_current_pos.append(np.array([1.0,0.25]))
                self.multi_current_pos.append(np.array([0.5,1.75]))
            self.multi_current_vel.append(np.zeros(2)) # initial velocity = [0,0]

        # update lasers
        self.update_lasers_isCollied_wrapper()
        ## multi_obs is list of agent_obs, state is multi_obs after flattenned
        multi_obs = self.get_multi_obs()
        return multi_obs

    # actions 为按智能体排列的二维加速度；依次更新速度、位置和障碍物。
    # 随后检测碰撞、计算奖励与终止标记、生成下一观测，返回三元组。
    def step(self,actions):
        last_d2target = []
        # print(actions)
        # time.sleep(0.1)
        for i in range(self.num_agents):

            pos = self.multi_current_pos[i]
            if i != self.num_agents - 1:
                pos_taget = self.multi_current_pos[-1]
                last_d2target.append(np.linalg.norm(pos-pos_taget))
            
            self.multi_current_vel[i][0] += actions[i][0] * self.time_step
            self.multi_current_vel[i][1] += actions[i][1] * self.time_step
            # 当前范数对整个 multi_current_vel 数组计算，并用来缩放当前智能体速度。
            # 它不是当前单机速度的范数，此处注释保留并说明现有实现。
            vel_magnitude = np.linalg.norm(self.multi_current_vel)
            if i != self.num_agents - 1:
                if vel_magnitude >= self.v_max:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max
            else:
                if vel_magnitude >= self.v_max_e:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max_e

            self.multi_current_pos[i][0] += self.multi_current_vel[i][0] * self.time_step
            self.multi_current_pos[i][1] += self.multi_current_vel[i][1] * self.time_step

        # Update obstacle positions
        for obs in self.obstacles:
            obs.position += obs.velocity * self.time_step
            # Check for boundary collisions and adjust velocities
            for dim in [0, 1]:
                if obs.position[dim] - obs.radius < 0:
                    obs.position[dim] = obs.radius
                    obs.velocity[dim] *= -1
                elif obs.position[dim] + obs.radius > self.length:
                    obs.position[dim] = self.length - obs.radius
                    obs.velocity[dim] *= -1

        # 先刷新激光与碰撞状态，再计算安全奖励，最后生成使用新激光数据的观测。
        Collided = self.update_lasers_isCollied_wrapper()
        rewards, dones= self.cal_rewards_dones(Collided,last_d2target)   
        multi_next_obs = self.get_multi_obs()
        # sequence above can't be disrupted

        return multi_next_obs, rewards, dones

    # 调试辅助：仅返回每个智能体归一化后的位置和速度，共四个分量。
    def test_multi_obs(self):
        total_obs = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            S_uavi = [
                pos[0]/self.length,
                pos[1]/self.length,
                vel[0]/self.v_max,
                vel[1]/self.v_max
            ]
            total_obs.append(S_uavi)
        return total_obs
    
    # 按智能体顺序生成局部观测；默认追捕者 26 维、目标 23 维。
    # 追捕者：自身 4 + 队友位置 4 + 激光 16 + 目标距离/方位 2。
    # 目标：自身 4 + 激光 16 + 到三名追捕者的距离 3。
    def get_multi_obs(self):
        total_obs = []
        single_obs = []
        S_evade_d = [] # dim 3 only for target
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            S_uavi = [
                pos[0]/self.length,
                pos[1]/self.length,
                vel[0]/self.v_max,
                vel[1]/self.v_max
            ] # dim 4
            S_team = [] # dim 4 for 3 agents 1 target
            S_target = [] # dim 2
            for j in range(self.num_agents):
                if j != i and j != self.num_agents - 1: 
                    pos_other = self.multi_current_pos[j]
                    S_team.extend([pos_other[0]/self.length,pos_other[1]/self.length])
                elif j == self.num_agents - 1:
                    pos_target = self.multi_current_pos[j]
                    d = np.linalg.norm(pos - pos_target)
                    theta = np.arctan2(pos_target[1]-pos[1], pos_target[0]-pos[0])
                    # 距离按标量 2*length 缩放，方位角保持弧度；激光距离直接使用场地长度单位。
                    S_target.extend([d/np.linalg.norm(2*self.length), theta])
                    if i != self.num_agents - 1:
                        S_evade_d.append(d/np.linalg.norm(2*self.length))

            S_obser = self.multi_current_lasers[i] # dim 16

            if i != self.num_agents - 1:
                single_obs = [S_uavi,S_team,S_obser,S_target]
            else:
                single_obs = [S_uavi,S_obser,S_evade_d]
            _single_obs = list(itertools.chain(*single_obs))
            total_obs.append(_single_obs)
            
        return total_obs

    # IsCollied 为碰撞标记，last_d 为本步运动前各追捕者与目标的距离。
    # 返回每个智能体的奖励数组及终止列表；捕获成功才在此处终止整个回合。
    def cal_rewards_dones(self,IsCollied,last_d):
        dones = [False] * self.num_agents
        rewards = np.zeros(self.num_agents)
        # 四项权重依次控制靠近目标、安全、围捕阶段及完成奖励。
        mu1 = 0.7 # r_near
        mu2 = 0.4 # r_safe
        mu3 = 0.01 # r_multi_stage
        mu4 = 5 # r_finish
        d_capture = 0.3
        d_limit = 0.75
        ## 1 reward for single rounding-up-UAVs:
        for i in range(3):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            pos_target = self.multi_current_pos[-1]
            v_i = np.linalg.norm(vel)
            dire_vec = pos_target - pos
            d = np.linalg.norm(dire_vec) # distance to target

            # 速度与目标方向的余弦衡量是否朝向目标；分母加 1e-3 避免静止时除零。
            cos_v_d = np.dot(vel,dire_vec)/(v_i*d + 1e-3)
            r_near = abs(2*v_i/self.v_max)*cos_v_d
            # r_near = min(abs(v_i/self.v_max)*1.0/(d + 1e-5),10)/5
            rewards[i] += mu1 * r_near # TODO: if not get nearer then receive negative reward
        
        ## 2 collision reward for all UAVs:
        for i in range(self.num_agents):
            if IsCollied[i]:
                r_safe = -10
            else:
                lasers = self.multi_current_lasers[i]
                # 取最近传感器读数计算安全项；即使量程内没有障碍，此公式也为负值。
                r_safe = (min(lasers) - self.L_sensor - 0.1)/self.L_sensor
            rewards[i] += mu2 * r_safe

        ## 3 multi-stage's reward for rounding-up-UAVs
        p0 = self.multi_current_pos[0]
        p1 = self.multi_current_pos[1]
        p2 = self.multi_current_pos[2]
        pe = self.multi_current_pos[-1]
        # 目标与追捕者组成的三个小三角形面积之和，与追捕者三角形面积比较。
        # 理论上面积相等对应目标位于三角形内部或边界；下方实现采用浮点精确相等判断。
        S1 = cal_triangle_S(p0,p1,pe)
        S2 = cal_triangle_S(p1,p2,pe)
        S3 = cal_triangle_S(p2,p0,pe)
        S4 = cal_triangle_S(p0,p1,p2)
        d1 = np.linalg.norm(p0-pe)
        d2 = np.linalg.norm(p1-pe)
        d3 = np.linalg.norm(p2-pe)
        Sum_S = S1 + S2 + S3
        Sum_d = d1 + d2 + d3
        Sum_last_d = sum(last_d)
        # 3.1 reward for target UAV:
        # 目标因总距离增加而获益；距离变化奖励被限制在 [-2, 2]。
        rewards[-1] += np.clip(10 * (Sum_d - Sum_last_d),-2,2)
        # print(rewards[-1])
        # 3.2 stage-1 track
        # 按面积及距离进入跟踪、包围、收缩阶段；各阶段的 0:2 切片仅奖励前两名追捕者。
        if Sum_S > S4 and Sum_d >= d_limit and all(d >= d_capture for d in [d1, d2, d3]):
            r_track = - Sum_d/max([d1,d2,d3])
            rewards[0:2] += mu3*r_track
        # 3.3 stage-2 encircle
        elif Sum_S > S4 and (Sum_d < d_limit or any(d >= d_capture for d in [d1, d2, d3])):
            r_encircle = -1/3*np.log(Sum_S - S4 + 1)
            rewards[0:2] += mu3*r_encircle
        # 3.4 stage-3 capture
        elif Sum_S == S4 and any(d > d_capture for d in [d1,d2,d3]):
            r_capture = np.exp((Sum_last_d - Sum_d)/(3*self.v_max))
            rewards[0:2] += mu3*r_capture
        
        ## 4 finish rewards
        # 面积相等且三段距离都不超过捕获阈值时，全体终止；完成奖励仍只加给索引 0、1。
        if Sum_S == S4 and all(d <= d_capture for d in [d1,d2,d3]):
            rewards[0:2] += mu4*10
            dones = [True] * self.num_agents

        return rewards,dones

    # 合并各障碍的逐束最短测距并返回各智能体碰撞标记，碰撞时清零速度。
    # 边界检测通过每次障碍测距间接进行；零障碍时此循环不会检查边界。
    def update_lasers_isCollied_wrapper(self):
        self.multi_current_lasers = []
        dones = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            current_lasers = [self.L_sensor] * self.num_lasers
            done_obs = []
            for obs in self.obstacles:
                obs_pos = obs.position
                r = obs.radius
                _current_lasers, done = update_lasers(pos,obs_pos,r,self.L_sensor,self.num_lasers,self.length)
                # 将每个圆障碍的结果逐束取最小值，得到整个场景的最近可见距离。
                current_lasers = [min(l, cl) for l, cl in zip(_current_lasers, current_lasers)]
                done_obs.append(done)
            done = any(done_obs)
            if done:
                self.multi_current_vel[i] = np.zeros(2)
            self.multi_current_lasers.append(current_lasers)
            dones.append(done)
        return dones

    # 在当前 Matplotlib 图中绘制轨迹、无人机和障碍，返回 RGBA 图像数组。
    # 每次调用都会追加当前位置到轨迹；UAV.png 按当前工作目录读取。
    def render(self):

        plt.clf()
        
        # load UAV icon
        uav_icon = mpimg.imread('UAV.png')
        # icon_height, icon_width, _ = uav_icon.shape

        # plot round-up-UAVs
        for i in range(self.num_agents - 1):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            self.history_positions[i].append(pos)
            trajectory = np.array(self.history_positions[i])
            # plot trajectory
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.3)
            # Calculate the angle of the velocity vector
            angle = np.arctan2(vel[1], vel[0])

            # plt.scatter(pos[0], pos[1], c='b', label='hunter')
            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            # plt.imshow(uav_icon, extent=(pos[0] - 0.05, pos[0] + 0.05, pos[1] - 0.05, pos[1] + 0.05))
            # plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(pos[0] - 0.05, pos[0] + 0.05, pos[1] - 0.05, pos[1] + 0.05))
            icon_size = 0.1  # Adjust this size to your icon's aspect ratio
            plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(-icon_size/2, icon_size/2, -icon_size/2, icon_size/2))

            # # Visualize laser rays for each UAV(can be closed when unneeded)
            # lasers = self.multi_current_lasers[i]
            # angles = np.linspace(0, 2 * np.pi, len(lasers), endpoint=False)
            
            # for angle, laser_length in zip(angles, lasers):
            #     laser_end = np.array(pos) + np.array([laser_length * np.cos(angle), laser_length * np.sin(angle)])
            #     plt.plot([pos[0], laser_end[0]], [pos[1], laser_end[1]], 'b-', alpha=0.2)

        # plot target
        plt.scatter(self.multi_current_pos[-1][0], self.multi_current_pos[-1][1], c='r', label='Target')
        self.history_positions[-1].append(copy.deepcopy(self.multi_current_pos[-1]))
        trajectory = np.array(self.history_positions[-1])
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.3)

        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle.position, obstacle.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)
        plt.xlim(-0.1, self.length+0.1)
        plt.ylim(-0.1, self.length+0.1)
        plt.draw()
        plt.legend()
        # plt.pause(0.01)
        # Save the current figure to a buffer
        # 使用 Agg 画布将当前图栅格化，以供训练入口导出 PNG 图像。
        canvas = agg.FigureCanvasAgg(plt.gcf())
        canvas.draw()
        buf = canvas.buffer_rgba()
        
        # Convert buffer to a NumPy array
        image = np.asarray(buf)
        return image

    # 绘制动画帧，以渐变色显示追捕轨迹；frame_num 为接口参数，函数体未使用。
    # 与 render 一样，每次绘图都会扩充历史轨迹。
    def render_anime(self, frame_num):
        plt.clf()
        
        uav_icon = mpimg.imread('UAV.png')

        for i in range(self.num_agents - 1):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            angle = np.arctan2(vel[1], vel[0])
            self.history_positions[i].append(pos)
            
            trajectory = np.array(self.history_positions[i])
            for j in range(len(trajectory) - 1):
                color = cm.viridis(j / len(trajectory))  # 使用 viridis colormap
                plt.plot(trajectory[j:j+2, 0], trajectory[j:j+2, 1], color=color, alpha=0.7)
            # plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=1)

            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            icon_size = 0.1
            plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(-icon_size/2, icon_size/2, -icon_size/2, icon_size/2))

        plt.scatter(self.multi_current_pos[-1][0], self.multi_current_pos[-1][1], c='r', label='Target')
        pos_e = copy.deepcopy(self.multi_current_pos[-1])
        self.history_positions[-1].append(pos_e)
        trajectory = np.array(self.history_positions[-1])
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.3)

        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle.position, obstacle.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)

        plt.xlim(-0.1, self.length + 0.1)
        plt.ylim(-0.1, self.length + 0.1)
        plt.draw()

    # 关闭当前 Matplotlib 图窗口。
    def close(self):
        plt.close()

# 圆形障碍物，具有中心位置、二维速度和半径。
class obstacle():
    # 在 length 对应的内部区域随机生成圆心和半径；当前速度为零，即静态障碍。
    def __init__(self, length=2):
        self.position = np.random.uniform(low=0.45, high=length-0.55, size=(2,))
        angle = np.random.uniform(0, 2 * np.pi)
        # speed = 0.03 
        speed = 0.00 # to make obstacle fixed
        self.velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])
        self.radius = np.random.uniform(0.1, 0.15)