import matplotlib.pyplot as plt
import numpy as np


def quatmultiply(a, b):
    """
    :param a: N * 4 matrix
    :param b: N * 4 matrix
    :return:
    """
    a = a.reshape((-1, 4))
    b = b.reshape((-1, 4))
    c = np.zeros((a.shape[0], 4), dtype=np.float)
    # scalar
    c[:, 0] = a[:, 0] * b[:, 0] - np.sum(a[:, 1:] * b[:, 1:], axis=1)
    c[:, 1:] = a[:, 0].reshape((-1, 1)) * b[:, 1:] + b[:, 0].reshape((-1, 1)) * a[:, 1:] + np.cross(a[:, 1:], b[:, 1:])
    return c


def quatconj(a):
    a = a.reshape((-1, 4))
    return np.array([a[:, 0], -1*a[:, 1], -1*a[:, 2], -1*a[:, 3]]).T


DIR_PATH = 'D:/pyProjects/ARprocessing/PAMAP2/Protocol/ahrs/subject101.dat'
label = 4
ACC_INX = [4, 5, 6]
GYRO_INX = [10, 11, 12]
MAG_INX = [13, 14, 15]
QUAR_INX = [16, 17, 18, 19]

draw_ratio = 50
data = np.loadtxt(DIR_PATH)
inx = np.argwhere(data[:, 1] == label)
acc = data[inx, ACC_INX]
quar = data[inx, QUAR_INX]
zeros = np.zeros((acc.shape[0], 1))
acc0 = np.hstack((zeros, acc))
acc_earth = quatmultiply(quatconj(quar), quatmultiply(acc0, quar))
N = acc.shape[0]
draw_acc = acc[N//2:N//2+N//draw_ratio, :]
draw_acc_earth = acc_earth[N//2:N//2+N//draw_ratio, 1:]
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(np.arange(N//draw_ratio), draw_acc[:, 0], label='--x--')
ax1.plot(np.arange(N//draw_ratio), draw_acc[:, 1], label='--y--')
ax1.plot(np.arange(N//draw_ratio), draw_acc[:, 2], label='--z--')
ax1.legend()
ax1.set_title('raw data')
ax2.plot(np.arange(N//draw_ratio), draw_acc_earth[:, 0], label='--x--')
ax2.plot(np.arange(N//draw_ratio), draw_acc_earth[:, 1], label='--y--')
ax2.plot(np.arange(N//draw_ratio), draw_acc_earth[:, 2], label='--z--')
ax2.legend()
ax2.set_title('to earth co-ordinate')

plt.figure(figsize=(8, 6))  # 设置画布大小
ax = plt.axes(projection='3d')  # 设置三维轴
ax.scatter3D(draw_acc[:, 0], draw_acc[:, 1], draw_acc[:, 2])  # 三个数组对应三个维度（三个数组中的数一一对应）
plt.xticks(range(11))  # 设置 x 轴坐标
plt.rcParams.update({'font.family': 'Times New Roman'})
plt.rcParams.update({'font.weight': 'normal'})
plt.rcParams.update({'font.size': 20})
plt.xlabel('X')
plt.ylabel('Y', rotation=38)  # y 轴名称旋转 38 度
ax.set_zlabel('Z')  # 因为 plt 不能设置 z 轴坐标轴名称，所以这里只能用 ax 轴来设置（当然，x 轴和 y 轴的坐标轴名称也可以用 ax 设置）


plt.figure(figsize=(8, 6))  # 设置画布大小
ax = plt.axes(projection='3d')  # 设置三维轴
ax.scatter3D(draw_acc_earth[:, 0], draw_acc_earth[:, 1], draw_acc_earth[:, 2])  # 三个数组对应三个维度（三个数组中的数一一对应）
plt.xticks(range(11))  # 设置 x 轴坐标
plt.rcParams.update({'font.family': 'Times New Roman'})
plt.rcParams.update({'font.weight': 'normal'})
plt.rcParams.update({'font.size': 20})
plt.xlabel('X')
plt.ylabel('Y', rotation=38)  # y 轴名称旋转 38 度
ax.set_zlabel('Z')  # 因为 plt 不能设置 z 轴坐标轴名称，所以这里只能用 ax 轴来设置（当然，x 轴和 y 轴的坐标轴名称也可以用 ax 设置）
plt.show()