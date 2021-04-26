import PAMAP2.preprocess_data as pp
import PAMAP2.FeatureExtract as fe
import pandas as pd
import numpy as np
from time import time
import os
"""
传统方法，特征提取过程
    载入原始数据 D:/pyProjects/ARprocessing/PAMAP2/Protocol/ahrs/subject1xx.dat
    数据预处理  (坐标变换/不变换)
    信号处理（中值滤波->频段选取,信号分离）
    保存数据 D:/pyProjects/ARprocessing/PAMAP2/Protocol/dsp/subject1xx.dat
           D:/pyProjects/ARprocessing/PAMAP2/Protocol/dsp_AccTransfer/subject1xx.dat
    
    载入数据
    for every windows:
        数据扩充（频谱数据，时域幅度数据，频域幅度数据）
        特征提取
        缓存特征和标签
    保存数据 D:/pyProjects/ARprocessing/PAMAP2/Protocol/fe/subject1xx.dat
           D:/pyProjects/ARprocessing/PAMAP2/Protocol/fe_AccTransfer/subject1xx.dat
    
特征：
t_axial: mean,std,median absolute deviation,max,min,interquartile range,sum of area,mean_energy,energy,pearsonr,skewness,kurtosis,4-order auto regression
t_magnitude: mean,std,median absolute deviation,max,min,interquartile range,sum of area,mean_energy,energy,skewness,kurtosis,4-order auto regression
f_axial: mean,std,median absolute deviation,max,min,interquartile range,sum of area,skewness, kurtosis, mean_energy, freq of max, mean freq, all bands energy
f_magnitude:mean,std,median absolute deviation,max,min,interquartile range,sum of area,skewness, kurtosis, mean_energy, freq of max, mean freq
"""
raw_data_dir = 'D:/pyProjects/ARprocessing/PAMAP2/Protocol/ahrs/'
dsp_data_dir = 'D:/pyProjects/ARprocessing/PAMAP2/Protocol/dsp_AccTransfer/'
fe_data_dir = 'D:/pyProjects/ARprocessing/PAMAP2/Protocol/fe_AccTransfer/'

# from raw data
t0 = time()
if not os.path.exists(dsp_data_dir):
    os.mkdir(dsp_data_dir)
convert = True
filenames = os.listdir(raw_data_dir)
for file in filenames:
    if file.endswith('.dat'):
        print(f'{file} is reading...')
        data = np.loadtxt(os.path.join(raw_data_dir, file))
        if convert:
            data = pp.to_cheat_coordinate(data, sen_list=[0])
        print(f'{file} is preprocessing...')
        data = pp.select_colums(data)
        data = pp.head_tail_delete(data, 1000)
        data = pp.discard_data(data, discard_label=0)
        data_x, data_y = pp.divide_x_y(data)
        data_y = pp.labels_adjust(data_y)
        data_x = np.array([pd.Series(i).interpolate() for i in data_x.T]).T

        print(f'{file} signal processing...')
        data_x = fe.median(data_x, kernel_size=5)
        acc = [0, 1, 2, 9, 10, 11]  # 计算 grav body body_jerk
        gyro = [3, 4, 5, 12, 13, 14]  # 计算 body body_jerk
        acc_grav = acc_body = acc_jerk = gyro_body = gyro_jerk = np.empty((len(data_x), 0))
        for i in acc:
            _, grav, body, _ = fe.components_selection_one_signal(data_x[:, i])
            acc_grav = np.hstack((acc_grav, np.array(grav).reshape(-1, 1)))
            acc_body = np.hstack((acc_body, np.array(body).reshape(-1, 1)))
            acc_jerk = np.hstack((acc_jerk, fe.jerk_one_signal(body).reshape(-1, 1)))

        for i in gyro:
            _, _, body, _ = fe.components_selection_one_signal(data_x[:, i])
            gyro_body = np.hstack((gyro_body, np.array(body).reshape(-1, 1)))
            gyro_jerk = np.hstack((gyro_jerk, fe.jerk_one_signal(body).reshape(-1, 1)))
        # 'hand->chest' then 'acc->gyro' then 'grav->body->jerk' then'x->y->z'
        data_dsp = np.hstack((acc_grav[:, 0:3], acc_body[:, 0:3], acc_jerk[:, 0:3], gyro_body[:, 0:3], \
                              gyro_jerk[:, 0:3], acc_grav[:, 3:6], acc_body[:, 3:6], acc_jerk[:, 3:6], \
                              gyro_body[:, 3:6], gyro_jerk[:, 3:6], data_y.reshape(-1, 1)))
        print(f'save file...')
        np.savetxt(os.path.join(dsp_data_dir, file), data_dsp, delimiter=' ')

print('raw data preprocessing and signal processing time: %.3fs' % float(time()-t0))

t0 = time()
window_length = 100
stride_len = 50
if not os.path.exists(fe_data_dir):
    os.mkdir(fe_data_dir)

# feature name generate, by the order: t, mag_t, f, mag_f
feature_names = fe.t_features_names() + fe.mag_t_features_names() + fe.f_features_names() \
                        + fe.mag_f_features_names()
print('total feature number :', len(feature_names))
with open(os.path.join(fe_data_dir, 'feature_names.txt'), 'w') as f:
    for name in feature_names:
        f.write(name + '\n')

filenames = os.listdir(dsp_data_dir)
for file in filenames:
    if file.endswith('.dat'):
        print(f'{file} is reading...')
        data = np.loadtxt(os.path.join(dsp_data_dir, file))
        data_x, data_y = data[:, 0:-1], data[:, -1]

        features = np.empty((0, len(feature_names)))
        labels = []
        # for every window
        print('extract features from every window...')
        number = (len(data_x) - window_length) // stride_len + 1
        for i in range(number):
            start = i * stride_len
            end = start + window_length

            # label
            data_y_w = data_y[start + window_length // 2]
            labels.append(data_y_w)

            # window data in triaxis time domain
            data_x_w = data_x[start:end, :]
            t_data = data_x_w
            f_data = fe.fast_fourier_transform(t_data)
            mag_t_data = fe.magnitude_of_triaxial(t_data)
            mag_f_data = fe.magnitude_of_triaxial(f_data)

            # features generate
            t_features = fe.t_features_generate(t_data)
            mag_t_features = fe.mag_t_features_generate(mag_t_data)
            f_features = fe.f_features_generate(f_data)
            mag_f_features = fe.mag_f_features_generate(mag_f_data)
            feature_vector = t_features + mag_t_features + f_features + mag_f_features

            # save feature_vector
            features = np.vstack((features, np.array(feature_vector).reshape((1, -1))))
        print('instances save...')
        instances = np.hstack((features, np.array(labels).reshape(-1, 1)))
        np.savetxt(os.path.join(fe_data_dir, file), instances, delimiter=' ')

print('feature extracting time: %.3fs' % float(time()-t0))




