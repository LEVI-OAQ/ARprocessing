"""
    信号可视化
    指定 subject ID 和 活动 ID 和 sensor ID，然后可视化
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
from PAMAP2.preprocess_data import select_colums, head_tail_delete, labels_adjust, discard_data, divide_x_y, to_cheat_coordinate
import warnings
warnings.filterwarnings('ignore')
DIR_PATH = 'D:/pyProjects/ARprocessing/PAMAP2/Protocol/ahrs/mahony/'
SR = 100
SEN_IDs = {
    'hand_acc': [0, 1, 2],
    'hand_gyro': [3, 4, 5],
    'hand_mag': [6, 7, 8],
    'chest_acc': [9, 10, 11],
    'chest_gyro': [12, 13, 14],
    'chest_mag': [15, 16, 17]
}
LABELS = ['lying', 'sitting', 'standing', 'walking', 'running', 'cycling', 'Nordic walking', 'ascending stairs',
          'descending stairs', 'vacuum cleaning', 'ironing', 'rope jumping']


def median(data, kernel_size=5):
    if len(data.shape) == 1:
        n_cols = 1
    else:
        n_cols = data.shape[1]
    for i in range(n_cols):

        data[:, i] = medfilt(data[:, i], kernel_size=kernel_size)
    return data


def signal_visualize(sub_id, act_id, sen_id, width, height, convert, transfer_mod):
    filename = 'subject10' + str(sub_id) + '.dat'
    data = np.loadtxt(DIR_PATH + filename)

    if convert:
        data = to_cheat_coordinate(data, sen_list=[0], transfer_mod=transfer_mod)   # coordinate convert, from hand-imu frame to chest-imu frame
    else:
        transfer_mod = 'no transfer'

    data = select_colums(data)
    data = head_tail_delete(data, 1000)
    data = discard_data(data, discard_label=0)
    data_x, data_y = divide_x_y(data)
    data_y = labels_adjust(data_y)

    if act_id == 'all':
        draw_data = data_x[:, SEN_IDs[sen_id]]
        draw_label = data_y
        title = 'subject: ' + str(sub_id) + ' activity: ' + str(act_id) + ' sensor: ' + sen_id + \
                ' Frame: ' + transfer_mod
    else:
        draw_data = data_x[data_y == act_id]
        draw_data = draw_data[:, SEN_IDs[sen_id]]
        draw_label = data_y[data_y == act_id]
        title = 'subject: ' + str(sub_id) + ' activity: ' + LABELS[act_id] + ' sensor: ' + sen_id + \
                ' Frame: ' + transfer_mod

    len_dd = len(draw_data)
    time = [1/float(SR)*i for i in range(len_dd)]
    fig = plt.figure(figsize=(width, height))
    # draw_data = median(draw_data, kernel_size=5)
    plt.plot(time, draw_data[:, 0], color='r', label=sen_id+'_X')
    plt.plot(time, draw_data[:, 1], color='g', label=sen_id+'_Y')
    plt.plot(time, draw_data[:, 2], color='b', label=sen_id+'_Z')
    if act_id == 'all':
        plt.plot(time, draw_label * 5, color='y', label='label')
    plt.ylabel(sen_id)
    plt.xlabel('Time (s)')
    plt.legend()
    plt.title(title)


# print('visualize...')
signal_visualize(1, 5, 'chest_acc', 12, 3, convert=False, transfer_mod='to_earth')
signal_visualize(1, 5, 'chest_acc', 12, 3, convert=True, transfer_mod='to_earth')
signal_visualize(1, 5, 'hand_acc', 12, 3, convert=False, transfer_mod='to_earth')
signal_visualize(1, 5, 'hand_acc', 12, 3, convert=True, transfer_mod='to_earth')
signal_visualize(1, 5, 'hand_acc', 12, 3, convert=True, transfer_mod='to_chest')





plt.show()

