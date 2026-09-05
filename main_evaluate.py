# 动画评估入口：加载模型、记录速度并展示单次围捕过程。
# 本文件注释描述当前实现；参数限制与特殊行为以具体代码为准。

from maddpg import MADDPG
from sim_env import UAVEnv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
warnings.filterwarnings('ignore')

# 用长度 window_size 的均匀窗口平滑数据；valid 模式只保留完整重叠位置。
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# 按时间绘制各智能体速率曲线；索引 3 在当前四智能体配置中标为目标。
def plot_velocity_magnitude(time_steps, velocities_magnitude):
    plt.figure(figsize=(15, 4))  
    for i in range(len(velocities_magnitude)):
        if i!=3:
            plt.plot(time_steps, velocities_magnitude[i], label=f'UAV {i}')
        else:
            plt.plot(time_steps, velocities_magnitude[i], label='Target')
    plt.xlabel("Time Steps")
    plt.ylabel("Magnitude")
    plt.title("UAV Velocity Magnitude")
    plt.legend()
    plt.grid(True)
    plt.show()

# 按时间绘制各智能体的水平速度分量。
def plot_velocity_x(time_steps, velocities_x):
    plt.figure(figsize=(15, 4))  
    for i in range(len(velocities_x)):
        if i!=3:
            plt.plot(time_steps, velocities_x[i], label=f'UAV {i}')
        else:
            plt.plot(time_steps, velocities_x[i], label='Target')
    plt.xlabel("Time Steps")
    plt.ylabel("$vel_x$")
    plt.title("UAV $Vel_x$")
    plt.legend()
    plt.grid(True)
    plt.show()

# 按时间绘制各智能体的竖直速度分量。
def plot_velocity_y(time_steps, velocities_y):
    plt.figure(figsize=(15, 4))  
    for i in range(len(velocities_y)):
        if i!=3:
            plt.plot(time_steps, velocities_y[i], label=f'UAV {i}')
        else:
            plt.plot(time_steps, velocities_y[i], label='Target')
    plt.xlabel("Time Steps")
    plt.ylabel("$vel_y$")
    plt.title("UAV $Vel_y$")
    plt.legend()
    plt.grid(True)
    plt.show()

# 在三个子图中展示速率、水平分量和竖直分量；输入按智能体分组。
def plot_velocities(velocities_magnitude, velocities_x, velocities_y):
    time_steps = range(len(velocities_magnitude[0]))
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    for i in range(len(velocities_magnitude)):
        if i != 3:
            axs[0].plot(time_steps, velocities_magnitude[i], label=f'UAV {i}')
        else:
            axs[0].plot(time_steps, velocities_magnitude[i], label=f'Target')
    axs[0].set_title('Speed Magnitude vs Time')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Speed Magnitude')
    axs[0].legend()

    for i in range(len(velocities_x)):
        if i != 3:
            axs[1].plot(time_steps, velocities_x[i], label=f'UAV {i}')
        else:
            axs[1].plot(time_steps, velocities_x[i], label=f'Target')
    axs[1].set_title('Velocity X Component vs Time')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Velocity X Component')
    axs[1].legend()

    for i in range(len(velocities_y)):
        if i != 3:
            axs[2].plot(time_steps, velocities_y[i], label=f'UAV {i}')
        else:
            axs[2].plot(time_steps, velocities_y[i], label=f'Target')
    axs[2].set_title('Velocity Y Component vs Time')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Velocity Y Component')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    env = UAVEnv()
    n_agents = env.num_agents
    n_actions = 2
    actor_dims = []
    velocities_magnitude = [[] for _ in range(env.num_agents)]  # record magnitude of vel
    velocities_x = [[] for _ in range(env.num_agents)]  # record vel_x
    velocities_y = [[] for _ in range(env.num_agents)]  # record vel_y

    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
                           fc1=128, fc2=128, alpha=0.0001, beta=0.003, scenario='UAV_Round_up',
                           chkpt_dir='tmp/maddpg/')
    
    # 网络尺寸及场景目录需要与已有权重一致。
    maddpg_agents.load_checkpoint()
    print('---- Evaluating ----')

    obs = env.reset()

    # 动画帧回调：先记录当前速度，再执行策略、推进环境并绘制新状态。
    # 任一终止标记为真时停止动画；返回空列表，当前动画未启用 blit。
    def update(frame):
        global obs,velocities_magnitude,velocities_x,velocities_y

        for i in range(env.num_agents):
            vel = env.multi_current_vel[i]
            v_x, v_y = vel
            speed = np.linalg.norm(vel)

            velocities_magnitude[i].append(speed)
            velocities_x[i].append(v_x)
            velocities_y[i].append(v_y)

        # 评估关闭探索噪声；本脚本 total_steps 保持为零，不影响确定性策略动作。
        actions = maddpg_agents.choose_action(obs, total_steps, evaluate=True)
        obs_, _, dones = env.step(actions)
        env.render_anime(frame)
        obs = obs_
        if any(dones):
            ani.event_source.stop()
            print("Round-up finished in",frame,"steps.")
            # smoothed_velocities_magnitude = [[] for _ in range(env.num_agents)]
            # smoothed_velocities_x = [[] for _ in range(env.num_agents)]  
            # smoothed_velocities_y = [[] for _ in range(env.num_agents)] 
            # for i in range(env.num_agents):
            #     _velocity_magnitude = moving_average(velocities_magnitude[i],window_size=5)
            #     _velocity_x = moving_average(velocities_x[i],window_size=5)
            #     _velocity_y = moving_average(velocities_y[i],window_size=5)
            #     smoothed_velocities_magnitude[i]=_velocity_magnitude
            #     smoothed_velocities_x[i]=_velocity_x
            #     smoothed_velocities_y[i]=_velocity_y
            # # plot_velocities(smoothed_velocities_magnitude,smoothed_velocities_x,smoothed_velocities_y)
            # time_steps = range(len(smoothed_velocities_magnitude[0]))
            # plot_velocity_magnitude(time_steps,smoothed_velocities_magnitude)
            # plot_velocity_x(time_steps, smoothed_velocities_x)
            # plot_velocity_y(time_steps, smoothed_velocities_y)
        return []

    total_steps = 0

    fig = plt.figure()
    # 最多调度 10000 帧，20 毫秒是显示调度间隔，物理步长由环境单独决定。
    ani = animation.FuncAnimation(fig, update, frames=10000, interval=20)
    plt.show()