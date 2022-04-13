import numpy as np
'''
  文件说明：
    使用NMS沿着梯度方向找到边缘像素的局部最大值
    - 输入 Mag：H x W 矩阵表示导数的大小
    - 输入 Ori：H x W 矩阵表示导数的方向
    - 输出 M：H x W 二值矩阵表示边缘
'''
def nonMaxSup(Mag, Ori, grad_Ori):
  # 初始化输出矩阵
  suppressed = np.copy(Mag)
  suppressed.fill(0)
  # 对每一个像素进行NMS
  for i in range(1, Mag.shape[0] - 1):
    for j in range(1, Mag.shape[1] - 1):
      # 获取当前像素的梯度方向
      cur_Ori = grad_Ori[i, j]
      # 获取当前像素的梯度大小
      cur_Mag = Mag[i, j]
      # 进行比较
      if cur_Ori == 0 or cur_Ori == 180:
        if cur_Mag > Mag[i, j-1] and cur_Mag > Mag[i, j+1]:
          suppressed[i, j] = 1
      elif cur_Ori == 45 or cur_Ori == -135:
        if cur_Mag > Mag[i-1, j+1] and cur_Mag > Mag[i+1, j-1]:
          suppressed[i, j] = 1
      elif cur_Ori == 90 or cur_Ori == -90:
        if cur_Mag > Mag[i-1, j] and cur_Mag > Mag[i+1, j]:
          suppressed[i, j] = 1
      elif cur_Ori == 135 or cur_Ori == -45:
        if cur_Mag > Mag[i-1, j-1] and cur_Mag > Mag[i+1, j+1]:
          suppressed[i, j] = 1
                  
  return suppressed

