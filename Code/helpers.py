import numpy as np

# 获得离散化的梯度方向，限定在8个方向
def get_gradient_angle(angleDeg):
    discrete = [0, 45, 90, 135, 180, -45, -90, -135, -180]
    dir = min(discrete, key=lambda x:abs(x-angleDeg))

    return dir


# 获得离散化的边缘梯度方向，限定在4个方向
def get_edge_angle(a):
    discrete = [0, 45, 90, 135, 180]
    dir = min(discrete, key=lambda x:abs(x-a))

    return dir


# 获得离散化的方向
def get_discrete_orientation(Ori):
    # 角度化梯度方向
    angle_Ori = np.degrees(Ori)
    # 并行运算，获取离散化的梯度方向
    get_gradient_angle_vect = np.vectorize(get_gradient_angle)
    discrete_gradient_orientation = get_gradient_angle_vect(angle_Ori)
    # 并行运算，获取离散化的边缘方向
    get_edge_angle_vect = np.vectorize(get_edge_angle)
    discrete_edge_orientation = get_edge_angle_vect(np.absolute(angle_Ori))

    return discrete_gradient_orientation, discrete_edge_orientation
