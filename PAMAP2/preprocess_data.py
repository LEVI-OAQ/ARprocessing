"""
    preprocess data from PAMAP2
    for every file:
    we need 2 step experiment
    first , chest and hand frame both transfer to reference frame
    second , hand frame transfer to reference frame and then transfer to chest frame continually


    , load data
    , discard data with label discard_label
    , select colums that we need
    , delete the head and tail samples(1000) of every activity
    , labels adjust
    , Colums are divided into features and labels

    , normalized (better solution ? )
    , stack together

    1，各个类在测试和训练中应该均衡分布
    2, 数据集由多个受试者采集，应该保证训练和测试，每个受试者都有（随机分）
    for every class
        get data_x_class_n, data_y_class_n
        sliding window
        randomly separate train data and test data based on ratio
        stack together
"""
import os
import numpy as np
import pickle



def labels_adjust(data_y):
    """
    – 1 lying           --->0
    – 2 sitting         --->1
    – 3 standing        --->2
    – 4 walking         --->3
    – 5 running         --->4
    – 6 cycling         --->5
    – 7 Nordic walking  --->6
    – 12 ascending stairs   --->7
    – 13 descending stairs  --->8
    – 16 vacuum cleaning    --->9
    – 17 ironing            --->10
    – 24 rope jumping       --->11
    :param data_y:
    :return:
    """
    data_y[data_y == 1] = 0
    data_y[data_y == 2] = 1
    data_y[data_y == 3] = 2
    data_y[data_y == 4] = 3
    data_y[data_y == 5] = 4
    data_y[data_y == 6] = 5
    data_y[data_y == 7] = 6
    data_y[data_y == 12] = 7
    data_y[data_y == 13] = 8
    data_y[data_y == 16] = 9
    data_y[data_y == 17] = 10
    data_y[data_y == 24] = 11
    return data_y


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


def to_cheat_coordinate(data, sen_list=[0, 1, 2], transfer_mod='to_chest'):
    """
    (0, x_chest, y_chest, z_chest) = Q_chest [Q_hand^-1 (0, x_hand, y_hand, z_hand) Q_hand] Q_chest^-1
    :param transfer_mod: to_chest or to_earth
    :param sen_list: [0, 1, 2] means all sensor convert to chest, 0-acc, 1:gyro, 2:mag
    :param data:
    :return:
    """
    # hand
    ACC_INX = [4, 5, 6]
    GYRO_INX = [10, 11, 12]
    MAG_INX = [13, 14, 15]
    QUAR_INX = [16, 17, 18, 19]
    INX_OFFSET = 17  # hand to chest
    zeros = np.zeros((data.shape[0], 1))
    inx = np.array([ACC_INX, GYRO_INX, MAG_INX])
    list_used = sen_list

    if transfer_mod == 'to_chest':
        for i in list_used:  # all
            hand_data = np.hstack((zeros, data[:, inx[i]]))
            on_world_coordinate = quatmultiply(quatconj(data[:, QUAR_INX]), quatmultiply(hand_data, data[:, QUAR_INX]))
            on_chest_coordinate = quatmultiply(
                quatmultiply(data[:, [INX_OFFSET + i for i in QUAR_INX]], on_world_coordinate),
                quatconj(data[:, [INX_OFFSET + i for i in QUAR_INX]]))
            data[:, inx[i]] = on_chest_coordinate[:, 1:]
            # hand_data = np.hstack((zeros, data[:, inx[i]]))
            # on_world_coordinate = quatmultiply(quatconj(data[:, QUAR_INX]), quatmultiply(hand_data, data[:, QUAR_INX]))
            # on_chest_coordinate = quatmultiply(quatconj(data[:, [INX_OFFSET + i for i in QUAR_INX]]), \
            #                                    quatmultiply(on_world_coordinate, data[:, [INX_OFFSET + i for i in QUAR_INX]]))
            # data[:, inx[i]] = on_chest_coordinate[:, 1:]
    elif transfer_mod == 'to_earth':
        for i in list_used:
            hand_data = np.hstack((zeros, data[:, inx[i]]))
            on_world_coordinate = quatmultiply(quatconj(data[:, QUAR_INX]), quatmultiply(hand_data, data[:, QUAR_INX]))
            temp = quatmultiply(quatmultiply(data[:, QUAR_INX], on_world_coordinate), quatconj(data[:, QUAR_INX]))

            data[:, inx[i]] = on_world_coordinate[:, 1:]

            chest_data = np.hstack((zeros, data[:, [j + INX_OFFSET for j in inx[i]]]))
            on_world_coordinate = quatmultiply(quatconj(data[:, [INX_OFFSET + j for j in QUAR_INX]]),\
                                               quatmultiply(chest_data, data[:, [INX_OFFSET + j for j in QUAR_INX]]))
            data[:, [j + INX_OFFSET for j in inx[i]]] = on_world_coordinate[:, 1:]
    return data


def discard_data(data, discard_label=0):
    """
        discard data with label=discard_label
    """
    if type(discard_label) == int:
        data = np.delete(data, np.argwhere(data[:, 0] == discard_label), axis=0)
    else:
        for label in discard_label:
            data = np.delete(data, np.argwhere(data[:, 0] == label), axis=0)
    return data


def head_tail_delete(data, n_delete):
    """ Function to delete head and tail samples(n_delete) for every label
    :param data: numpy.ndarray
        '0   :AID'
        '1-6 : IMU_hand'
        '7-12: IMU_chest'

    :param n_delete: int
        number of samples need to delete

    :return:
        Processed  data
    """
    last_label = data[0, 0]
    cnt = 0
    start_r = 0
    delete_r = np.empty(0, dtype=np.int)
    for r in np.arange(data.shape[0]):
        if last_label != data[r, 0]:
            if cnt < 2 * n_delete:
                delete_r = np.concatenate((delete_r, np.arange(start_r, r)))
            else:
                delete_r = np.concatenate((delete_r, np.arange(start_r, start_r + n_delete)))
                delete_r = np.concatenate((delete_r, np.arange(r - n_delete, r)))
            last_label = data[r, 0]
            start_r = r
        cnt += 1
    return np.delete(data, delete_r, axis=0)


# – 1 timestamp (s)
# – 2 activityID (see II.2. for the mapping to the activities)
# – 3 heart rate (bpm)
# – 4-20 IMU hand
# – 21-37 IMU chest
# – 38-54 IMU ankle

# – 1 temperature (°C)
# – 2-4 3D-acceleration data (ms -2 ), scale: ±16g, resolution: 13-bit
# – 5-7 3D-acceleration data (ms -2 ), scale: ±6g, resolution: 13-bit *
# – 8-10 3D-gyroscope data (rad/s)
# – 11-13 3D-magnetometer data (μT)
# – 14-17 orientation (invalid in this data collection)
def select_colums(data):
    """
    :param data: numpy.ndarray
    :param used_colums:
    :return: numpy.ndarray
        '0   :AID'
        '1-6 : IMU_hand'
        '7-12: IMU_chest'
    """
    # the colums we not used
    delete_colums = np.array([0, 2])
    IMU_hand = np.concatenate((np.array([3, ]), np.arange(7, 10), np.arange(16, 20)))
    IMU_chest = np.concatenate((np.array([20, ]), np.arange(24, 27), np.arange(33, 37)))
    IMU_ankle = np.arange(37, 54)
    delete_colums = np.concatenate([delete_colums, IMU_hand, IMU_chest, IMU_ankle])
    return np.delete(data, delete_colums, 1)


def divide_x_y(data):
    """
    :param data: numpy.ndarray
    :return: data_x, data_y
    """
    data_y = data[:, 0].astype(np.int)
    data_x = data[:, 1:]
    return data_x, data_y


def normalize(data_x):
    max_list = np.max(data_x, axis=0)
    min_list = np.min(data_x, axis=0)
    data_x = (data_x - min_list) / (max_list - min_list)

    # data_x = (data_x - np.mean(data_x, axis=0)) / np.std(data_x, axis=0)
    return data_x


def sliding_window(data_x, data_y, window_width, stride_len):
    number = (len(data_x) - window_width) // stride_len + 1
    print(number)
    x = np.empty((0, data_x.shape[1]))
    y = np.empty(0)
    for i in range(number):
        start = i * stride_len
        end = start + window_width
        data_x_t = data_x[start:end, :]
        data_y_t = (data_y[start + window_width//2],)
        x = np.vstack((x, data_x_t))
        y = np.concatenate((y, data_y_t))
    return x.reshape(number, window_width, -1), y


def data_preprocess(filename, convert=False):
    data = np.loadtxt(filename)
    if convert:
        data = to_cheat_coordinate(data, sen_list=[0])  # coordinate convert, from hand-imu frame to chest-imu frame
    data = select_colums(data)
    data = head_tail_delete(data, 1000)
    data = discard_data(data, discard_label=0)
    data_x, data_y = divide_x_y(data)
    data_y = labels_adjust(data_y)
    # use data from arhs, no need to  interpolate again
    # data_x = np.array([pd.Series(i).interpolate() for i in data_x.T]).T
    data_x = normalize(data_x)
    return data_x, data_y


def data_generate(data_dir, target_filename, window_width, stride_len):
    """Function to read the PAMAP2  raw data and process all sensor channels
    :param stride_len:
    :param window_width:
    :param data_dir: string
        path of the data folder
    :param target_filename: string
        save-path of preprocessed data
    :param n_class: int
        number of classes
    :param ratio: float
        Ratio of training data
    """
    file_list = os.listdir(data_dir)
    sensor_channels = 18

    # preprocess and load data to data_x, data_y
    data_x = np.empty((0, window_width, sensor_channels))
    data_y = np.empty(0)
    data_group = np.empty(0)
    for file in file_list:
        if file.endswith('.dat'):
            print(f'{file} is reading...')
            x, y = data_preprocess(os.path.join(data_dir, file), convert=False)
            group = int(file[-5])
            print(f'labels : {len(set(y))}, {set(y)}')
            x_slided, y_slided = sliding_window(x, y, window_width, stride_len)
            data_group = np.concatenate((data_group, np.ones(len(y_slided)) * group))
            data_x = np.vstack((data_x, x_slided))
            data_y = np.concatenate((data_y, y_slided))

    obj = (data_x, data_y, data_group)
    with open(os.path.join(data_dir, target_filename), 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    # # sliding window and separate train/test data, for every single label, train/total = ratio.
    # x_train = x_test = np.empty((0, window_width, sensor_channels))
    # y_train = y_test = np.empty(0)
    # print("sliding wi ndow and separate train data & test data...")
    # print(f'window width:{window_width}, stride_len:{stride_len}, train:total:{ratio}')
    # for label in np.arange(n_class):
    #     print(f'the data with label {label} is sliding and separating... ')
    #     index = np.argwhere(data_y == label)
    #     index = index.squeeze()
    #     data_x_l = data_x[index, :]
    #     data_y_l = data_y[index]
    #     x_slided, y_slided = sliding_window(data_x_l, data_y_l, window_width, stride_len)
    #     # random pick
    #     number = len(x_slided)
    #     sample_index = list(range(number))
    #     sample_index = np.array(random.sample(sample_index, int(ratio * number)))
    #     x_train = np.vstack((x_train, x_slided[sample_index, :]))
    #     y_train = np.concatenate((y_train, y_slided[sample_index]))
    #     x_test = np.vstack((x_test, np.delete(x_slided, sample_index, axis=0)))
    #     y_test = np.concatenate((y_test, np.delete(y_slided, sample_index)))

# the data has been interpolated and the quarternion has been calculated



