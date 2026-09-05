# 二维几何工具：圆形障碍物/方形边界测距与三角形面积计算。
# 本文件注释描述当前实现；参数限制与特殊行为以具体代码为准。

import numpy as np
import math

# 1 simulate lidar
# 计算位置 pos 对单个圆形障碍及方形边界的 num_lasers 束等角度测距。
# obs_pos/r 为圆心/半径，L 为最大量程，bound 为 [0, bound] 方形边长。
# 返回距离列表及碰撞标记；位于圆内或边界外时，各束距离返回零。
def update_lasers(pos, obs_pos, r, L, num_lasers, bound):

    distance_to_obs = np.linalg.norm(np.array(pos) - np.array(obs_pos))
    isInObs = distance_to_obs < r \
                or pos[0] < 0 \
                or pos[0] > bound \
                or pos[1] < 0 \
                or pos[1] > bound
    
    if isInObs:
        return [0.0] * num_lasers, isInObs
    
    # 覆盖完整圆周且不重复 2π 方向；先计算圆交点，再用墙面距离截短。
    angles = np.linspace(0, 2 * np.pi, num_lasers, endpoint=False)
    laser_lengths = [L] * num_lasers
    
    for i, angle in enumerate(angles):
        intersection_dist = check_obs_intersection(pos, angle, obs_pos, r, L)
        if laser_lengths[i] > intersection_dist:
            laser_lengths[i] = intersection_dist
    
    for i, angle in enumerate(angles):
        wall_dist = check_wall_intersection(pos, angle, bound, L)
        if laser_lengths[i] > wall_dist:
            laser_lengths[i] = wall_dist
    
    return laser_lengths, isInObs

# 从 start_pos 沿 angle（弧度）发射长度 max_distance 的线段，求与圆的最近交点距离。
# obs_pos/r 为圆心/半径；没有有效交点时返回最大量程。
def check_obs_intersection(start_pos, angle, obs_pos,r,max_distance):
    ox = obs_pos[0]
    oy = obs_pos[1]

    end_x = start_pos[0] + max_distance * np.cos(angle)
    end_y = start_pos[1] + max_distance * np.sin(angle)

    dx = end_x - start_pos[0]
    dy = end_y - start_pos[1]
    fx = start_pos[0] - ox
    fy = start_pos[1] - oy

    # 将线段 P(t)=start+t*(end-start) 代入圆方程，得到关于 t 的二次方程。
    a = dx**2 + dy**2
    b = 2 * (fx * dx + fy * dy)
    c = (fx**2 + fy**2) - r**2

    discriminant = b**2 - 4 * a * c

    if discriminant >= 0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)
        
        # 只接受线段范围内的根；先检查较小的根以选取最近交点。
        if 0 <= t1 <= 1:
            return t1 * max_distance
        if 0 <= t2 <= 1:
            return t2 * max_distance

    return max_distance

# 求射线与 [0, bound] 方形四条边的距离，返回不超过 L 的最近距离。
def check_wall_intersection(start_pos, angle, bound, L):

    # 只检测射线指向的墙；方向分量为零时跳过对应除法。
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    L_ = L
    #  (y = bound)
    if sin_theta > 0:  
        L_ = min(L_, abs((bound - start_pos[1]) / sin_theta))
    
    #  (y = 0)
    if sin_theta < 0:  
        L_ = min(L_, abs(start_pos[1] / -sin_theta))

    #  (x = bound)
    if cos_theta > 0: 
        L_ = min(L_, abs((bound - start_pos[0]) / cos_theta))
    
    #  (x = 0)
    if cos_theta < 0: 
        L_ = min(L_, abs(start_pos[0] / -cos_theta))

    return L_

# 返回三个二维点的无符号三角形面积，绝对误差 1e-9 内的零面积统一返回 0.0。
def cal_triangle_S(p1, p2, p3):
    # 二维叉积的绝对值除以二；取绝对值使顶点顺序不影响面积。
    S = abs(0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])))
    if math.isclose(S, 0.0, abs_tol=1e-9):
        return 0.0
    else:
        return S