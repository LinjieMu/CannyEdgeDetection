# Canny Edge Detection

## 1. 算法流程

1. 利用高斯梯度算子对图像进行**滤波**。

2. 根据滤波结果计算每一像素点的**边缘强度**。

   1. 使用卷积运算计算出图像x方向和y方向的梯度。

      计算x方向的梯度卷积核(举例)：
      
      <img src="https://tva1.sinaimg.cn/large/e6c9d24ely1h181yqd5w4j205c02cdfl.jpg" alt="image-20220413135723960" style="zoom:50%;" />
      
      计算y方向梯度的卷积核(举例)：
      
      <img src="https://tva1.sinaimg.cn/large/e6c9d24ely1h181xzhtsaj203m032gld.jpg" alt="image-20220413135644302" style="zoom:50%;" />
      
   2. 对梯度使用高斯卷积核进行平滑。
   
      注意：由卷积运算的结合律，也可以先对高斯卷积核进行梯度计算再做卷积运算。
   
   3. 边缘强度计算。
      
      ![](https://latex.codecogs.com/svg.latex?I_m=\sqrt{dx^2+dy^2})
   
3. 计算**边缘方向**。

   梯度方向为：
   
   <img src="https://tva1.sinaimg.cn/large/e6c9d24ely1h181b6tsamj20a003cjr9.jpg" alt="image-20220413133449289" style="zoom:25%;" />
   
   边缘方向：
   
   <img src="https://tva1.sinaimg.cn/large/e6c9d24ely1h181awtsjbj20b203iq2t.jpg" alt="image-20220413133433522" style="zoom: 25%;" />
   
4. **检测局部最大值**。

   选择**沿梯度方向**最大的点作为局部最大值点。由于梯度方向上的点可能不落在像素格内，因此限定梯度方向在
   
   ![](https://latex.codecogs.com/svg.latex?[-3\pi/4,-\pi/2,-\pi/4,0,\pi/4,\pi/2,3\pi/4,\pi])
   
   上。使用8邻域法比较相邻的几个像素点即可。

5. 连接生成边缘.

   - 沿着**边缘方向**进行边缘点的连接。

   - 哪些点可以生长进来？

     我们使用双阈值法，大于大阈值的点一定是边缘点，位于双阈值之间的点可能是，根据链接情况选择，小于小阈值的点一定不是边缘点。

   - 从哪些点开始生长？

     从肯定是边缘点的点(大于大阈值)开始。

## 2. 代码说明

1. `findDerivatives.py`文件。

   在该文件中我们对图像进行高斯平滑，并分别计算出x和y方向上的梯度。

   ```python
   # 计算x方向的梯度并对梯度平滑
   Ix = signal.convolve2d(I_gray, dx, mode='same', boundary='symm')
   Magx = signal.convolve2d(Ix, gaussian, mode='same', boundary='symm')
   # 计算y方向的梯度并对梯度平滑
   Iy = signal.convolve2d(I_gray, dy, mode='same', boundary='symm')
   ```

   进而计算出，边缘强度和梯度方向。

   ```python
   # 计算边缘强度
   Mag = np.sqrt(Magx**2 + Magy**2)
   # 计算梯度的方向
   Ori = np.arctan2(Magy, Magx)
   ```

2. `nonMaxSup.py`文件。

   使用NMS沿着离散化后梯度方向找到边缘像素的局部最大值。使用八邻域法。

   ```python
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
   ```

## 3. 效果展示

![3096](https://tva1.sinaimg.cn/large/e6c9d24ely1h180qa0dn6j20dd08xaa5.jpg)

![3096_Result](https://tva1.sinaimg.cn/large/e6c9d24ely1h180qwz2u1j20dd08xwev.jpg)

![22013](https://tva1.sinaimg.cn/large/e6c9d24ely1h180re7fg8j208x0ddmxy.jpg)

![22013_Result](https://tva1.sinaimg.cn/large/e6c9d24ely1h180rn6l7zj208x0dd759.jpg)

## 4. 使用方法

- 到指定文件夹下，下载该文件

  ```bash
  git clone https://github.com/LinjieMu/CannyEdgeDetection.git
  ```

- 确保本地环境为python3.7，如果不是，请使用conda创建虚拟环境

- 进入文件目录下安装对应依赖库

  ```bash
  pip install -r requirements.txt
  ```

  如果在大陆您的下载过慢，可尝试使用清华源

  ```bash
  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
  ```

- 将目标文件放入Images文件夹中，执行如下命令(如果您的操作环境是Linux/Macos)

  ```bash
  python3 Code/cannyEdge.py --image_folder Images --save_folder Results
  ```

  如果你的操作环境是Windows，执行下面命令

  ```bash
  python Code//cannyEdge.py --image_folder Images --save_folder Results

- 此时即可在Results文件夹下看到结果

