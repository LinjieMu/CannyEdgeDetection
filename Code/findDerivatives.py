import numpy as np
from scipy import signal
import utils
import nonMaxSup

'''
  文件说明：
    计算灰度图像的梯度放置信息
     - 输入 I_gray：H x W 矩阵作为图像
     - 输出 Mag：H x W 矩阵表示导数的大小
     - 输出 Magx：H x W 矩阵表示沿 x 轴的导数
     - 输出 Magy：H x W 矩阵表示沿 y 轴的导数
     - 输出 Ori：H x W 矩阵表示导数的方向
'''

def findDerivatives(I_gray):
    # 高斯平滑核
    gaussian = np.array([
      [2, 4, 5, 4, 2], 
      [4, 9, 12, 9, 4], 
      [5, 12, 15, 12, 5], 
      [4, 9, 12, 9, 4], 
      [2, 4, 5, 4, 2]]) / 159.0   
    # 梯度计算核
    dx = np.asarray([
      [-1.0, 0.0, 1.0], 
      [-2.0, 0.0, 2.0], 
      [-1.0, 0.0, 1.0]
      ])
    dy = np.asarray([
      [1.0, 2.0, 1.0], 
      [0.0, 0.0, 0.0], 
      [-1.0, -2.0, -1.0]
      ])
    # 计算x方向的梯度并对梯度平滑
    Ix = signal.convolve2d(I_gray, dx, mode='same', boundary='symm')
    Magx = signal.convolve2d(Ix, gaussian, mode='same', boundary='symm')
    # 计算y方向的梯度并对梯度平滑
    Iy = signal.convolve2d(I_gray, dy, mode='same', boundary='symm')
    Magy = signal.convolve2d(Iy, gaussian, mode='same', boundary='symm')
    # 计算边缘强度
    Mag = np.sqrt(Magx**2 + Magy**2)
    # 计算梯度的方向
    Ori = np.arctan2(Magy, Magx)
    return Mag, Magx, Magy, Ori