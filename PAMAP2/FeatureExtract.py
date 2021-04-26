import numpy as np
import scipy as sp
from scipy.signal import medfilt
from scipy import fftpack
from statsmodels.robust import mad as median_deviation  # import the median deviation function
from scipy.stats import iqr, pearsonr
from spectrum import burg as bg
import math
import matplotlib.pyplot as plt

Sample_freq = 100


def median(data, kernel_size=5):
    if len(data.shape) == 1:
        n_cols = 1
    else:
        n_cols = data.shape[1]
    for i in range(n_cols):
        data[:, i] = medfilt(data[:, i], kernel_size=kernel_size)
    return data


def components_selection_one_signal(t_signal):
    """
    :param t_signal: 1D numpy array (time domain signal)
    :return: (total_component, t_DC_component , t_body_component, t_noise)

    """
    # the cutoff frequency between the DC components [0,0.3]Hz and the body components[0.3,20]Hz and the high frequency
    # noise components [20,25]Hz
    freq1 = 0.3
    freq2 = 20

    t_signal = np.array(t_signal)
    t_signal_length = len(t_signal)

    f_signal = fftpack.fft(t_signal)

    # generate frequencies associated to f_signal complex values [-50,50]Hz
    freqs = np.array(sp.fftpack.fftfreq(t_signal_length, d=1 / float(Sample_freq)))

    # DC_component: f_signal values having freq between [-0.3 hz to 0 hz] and from [0 hz to 0.3hz]
    #               (-0.3 and 0.3 are included)

    # noise components: f_signal values having freq between [-50 hz to -20 hz] and from [20 hz to 50 hz]
    #                   (-50 and 50 hz included, -20hz and 20hz not included)

    # selecting body_component: f_signal values having freq between [-20 hz to -0.3 hz] and from [0.3 hz to 20 hz]
    #                           (-0.3 and 0.3 not included , -20hz and 20hz included)

    f_DC_signal = []
    f_body_signal = []
    f_noise_signal = []

    for i in range(len(freqs)):

        freq = freqs[i]

        # selecting the f_signal value associated to freq
        value = f_signal[i]

        # Selecting DC_component values
        if abs(freq) > 0.3:
            f_DC_signal.append(float(0))
        else:
            f_DC_signal.append(value)

        # Selecting noise component values
        if abs(freq) <= 20:
            f_noise_signal.append(float(0))
        else:
            f_noise_signal.append(value)

        # Selecting body_component values
        if abs(freq) <= 0.3 or abs(freq) > 20:
            f_body_signal.append(float(0))
        else:
            f_body_signal.append(value)

    # applying the inverse fft to signals in freq domain and put them in float format
    t_DC_component = fftpack.ifft(np.array(f_DC_signal)).real
    t_body_component = fftpack.ifft(np.array(f_body_signal)).real
    t_noise = fftpack.ifft(np.array(f_noise_signal)).real

    total_component = t_signal - t_noise  # extracting the total component(filtered from noise)

    return total_component, t_DC_component, t_body_component, t_noise


def visualize_signal(signal, x_labels, y_labels, title, legend):
    """
    :param signal: 1D column
    :param x_labels: the X axis info (figure)
    :param y_labels: the Y axis info (figure)
    :param title: figure's title
    :param legend: figure's legend
    :return:
    """

    # Define the figure's dimensions
    plt.figure(figsize=(12, 4))

    # convert row numbers in time durations
    time = [1 / float(Sample_freq) * i for i in range(len(signal))]

    # plotting the signal
    plt.plot(time, signal, label=legend)  # plot the signal and add the legend

    plt.xlabel(x_labels)  # set the label of x axis in the figure
    plt.ylabel(y_labels)  # set the label of y axis in the figure
    plt.title(title)  # set the title of the figure
    plt.legend(loc="upper left")  # set the legend in the upper left corner
    plt.show()  # show the figure


def verify_gravity(accXYZ):

    # apply the filtering method to acc_[X,Y,Z] and store gravity components
    grav_acc_X = components_selection_one_signal(accXYZ[:, 0])[1]
    grav_acc_Y = components_selection_one_signal(accXYZ[:, 1])[1]
    grav_acc_Z = components_selection_one_signal(accXYZ[:, 2])[1]

    # calculating gravity magnitude signal
    grav_acc_mag = [math.sqrt((grav_acc_X[i]**2 + grav_acc_Y[i]**2 + grav_acc_Z[i]**2)) for i in range(len(grav_acc_X))]

    x_labels = 'time in seconds'
    y_labels = 'gravity amplitude in 1g'
    title = 'the magnitude of gravity'
    legend = 'gravity'

    visualize_signal(grav_acc_mag, x_labels, y_labels, title, legend)  # visualize gravity magnitude signal
    print('mean value = ' + str(np.array(grav_acc_mag).mean())[0:5] + ' g')  # print the gravity magnitude mean value


def jerk_one_signal(signal):
    dt = 1.0 / Sample_freq
    jerk = [(signal[i + 1] - signal[i]) / dt for i in range(len(signal)-1)]
    jerk.append(jerk[-1])
    return np.array(jerk)


def fast_fourier_transform(data):
    complex_f = fftpack.fft(data, axis=0)
    amplitude_f = np.abs(complex_f)
    return amplitude_f


def magnitude_of_triaxial(data):
    mag = np.empty((len(data), 0))
    for i in range(data.shape[1] // 3):
        mag = np.hstack((mag, np.sqrt(np.sum(np.power(data[:, 3*i:3*i+3], 2), axis=1)).reshape(-1, 1)))
    return mag


def mean_axial(data):
    return list(np.mean(data, axis=0))


def std_axial(data):
    return list(np.std(data, axis=0))


def mad_axial(data):
    return list(median_deviation(data, axis=0))  # calculate the median absolute deviation value of each column


def max_axial(data):
    return list(np.max(data, axis=0))


def min_axial(data):
    return list(np.min(data, axis=0))


def iqr_axial(data):
    return list(iqr(data, axis=0))  # calculate the interquartile range value of each column


# def entropy_axial(data):
#     return entropy(abs(data), axis=0)

def sma_axial(data):
    return list(np.sum(abs(data), axis=0))


def energy_axial(data):
    return list(np.sum(np.power(data, 2), axis=0))


def mean_energy_axial(data):
    return list(np.sum(np.power(data, 2), axis=0) / len(data))


def pearsonr_axial(data):
    if len(data.shape) == 1:
        n_col = 1
    else:
        n_col = data.shape[1]
    results = []
    for i in range(n_col // 3):
        results.append(pearsonr(data[:, 3*i], data[:, 3*i+1])[0])
        results.append(pearsonr(data[:, 3*i+1], data[:, 3*i+2])[0])
        results.append(pearsonr(data[:, 3*i+2], data[:, 3*i+1])[0])
    return results


def skewness_axial(data):
    return list(sp.stats.skew(data, axis=0))


def kurtosis_axial(data):
    return list(sp.stats.kurtosis(data, axis=0))


def arburg_axial(data):
    if len(data.shape) == 1:
        n_col = 1
    else:
        n_col = data.shape[1]
    ar_vector = []
    for i in range(n_col):
        ar_vector = ar_vector + list(bg._arburg2(data[:, i], 4)[0][1:].real)
    return ar_vector


def max_freq_axial(data):
    if len(data.shape) == 1:
        n_col = 1
    else:
        n_col = data.shape[1]
    results = []
    freqs = sp.fftpack.fftfreq(len(data), d=1/float(Sample_freq))
    for i in range(n_col):
        results.append(freqs[data[:, i].argmax()])
    return results


def f_mean_freq_axial(data):
    freqs = sp.fftpack.fftfreq(len(data), d=1 / float(Sample_freq))
    return list(np.sum(data * np.array(freqs).reshape((-1, 1)), axis=0) / np.sum(data, axis=0))


# B1 = [(1, 9), (9, 17), (17, 25), (25, 33), (33, 41), (41, 49), (49, 57), (57, 65)]
# B2 = [(1, 17), (17, 31), (31, 49), (49, 65)]
# B3 = [(1, 25), (25, 49)]
#
#
# def f_bands_energy_axial(data):
#     band1_energy = [mean_energy_axial(data[tu[0]: tu[1], :])for tu in B1]
#     band2_energy = [mean_energy_axial(data[tu[0]: tu[1], :])for tu in B2]
#     band3_energy = [mean_energy_axial(data[tu[0]: tu[1], :])for tu in B3]


def t_features_names():

    Feature = ['Mean', 'Std', 'Mad', 'Max', 'Min', 'Iqr', 'Sma', 'MeanEnergy', 'Energy',
               'Pearsonr', 'Skewness', 'Kurtosis', 'Ar']
    Position = ['Hand', 'Chest']
    Sensor = ['Acc', 'Gyro']
    Signal = ['Grav', 'Body', 'Jerk']
    Axis = ['X', 'Y', 'Z']

    AR_id = ['_1', '_2', '_3', '_4']

    features_names = []
    feature_name = 't_'
    # 't_FeaturePositionSensorSignalAxisAR_id'
    for feature in Feature:
        temp_0 = feature_name
        feature_name = feature_name + feature
        for position in Position:
            temp_1 = feature_name
            feature_name = feature_name + position
            for sensor in Sensor:
                temp_2 = feature_name
                feature_name = feature_name + sensor
                if sensor == 'Gyro':
                    tp_Signal = Signal[1:]
                else:
                    tp_Signal = Signal

                for signal in tp_Signal:
                    temp_3 = feature_name
                    feature_name = feature_name + signal
                    for axis in Axis:
                        temp_4 = feature_name
                        feature_name = feature_name + axis
                        if feature == 'Ar':
                            for id in AR_id:
                                temp_5 = feature_name
                                feature_name = feature_name + id
                                features_names.append(feature_name)
                                feature_name = temp_5
                        else:
                            features_names.append(feature_name)
                        feature_name = temp_4
                    feature_name = temp_3
                feature_name = temp_2
            feature_name = temp_1
        feature_name = temp_0

    return features_names


# t_magnitude: mean,std,median absolute deviation,max,min,interquartile range,sum of area,mean_energy,
#     energy,skewness,kurtosis,4-order auto regression

def mag_t_features_names():
    # 'mag_t_FeaturePositionSensorSignalAR_id'
    Feature = ['Mean', 'Std', 'Mad', 'Max', 'Min', 'Iqr', 'Sma', 'Mean_energy', 'Energy',
               'Skewness', 'Kurtosis', 'Ar']
    Position = ['Hand', 'Chest']
    Sensor = ['Acc', 'Gyro']
    Signal = ['Grav', 'Body', 'Jerk']
    AR_id = ['_1', '_2', '_3', '_4']

    features_names = []
    feature_name = 'mag_t_'
    for feature in Feature:
        temp_0 = feature_name
        feature_name = feature_name + feature
        for position in Position:
            temp_1 = feature_name
            feature_name = feature_name + position
            for sensor in Sensor:
                temp_2 = feature_name
                feature_name = feature_name + sensor
                if sensor == 'Gyro':
                    tp_Signal = Signal[1:]
                else:
                    tp_Signal = Signal
                for signal in tp_Signal:
                    temp_3 = feature_name
                    feature_name = feature_name + signal
                    if feature == 'Ar':
                        for id in AR_id:
                            temp_4 = feature_name
                            feature_name = feature_name + id
                            features_names.append(feature_name)
                            feature_name = temp_4
                    else:
                        features_names.append(feature_name)
                    feature_name = temp_3
                feature_name = temp_2
            feature_name = temp_1
        feature_name = temp_0
    return features_names


def f_features_names():
    # mean,std,median absolute deviation,max,min,interquartile range,sum of area,skewness, kurtosis,
    # mean_energy, freq of max, mean freq, all bands energy
    Feature = ['Mean', 'Std', 'Mad', 'Max', 'Min', 'Iqr', 'Sma', 'Skewness', 'Kurtosis', 'MeanEnergy',
               'FreqOfMax', 'MeanFreq']
    Position = ['Hand', 'Chest']
    Sensor = ['Acc', 'Gyro']
    Signal = ['Grav', 'Body', 'Jerk']
    Axis = ['X', 'Y', 'Z']

    features_names = []
    feature_name = 'f_'
    for feature in Feature:
        temp_0 = feature_name
        feature_name = feature_name + feature
        for position in Position:
            temp_1 = feature_name
            feature_name = feature_name + position
            for sensor in Sensor:
                temp_2 = feature_name
                feature_name = feature_name + sensor
                if sensor == 'Gyro':
                    tp_Signal = Signal[1:]
                else:
                    tp_Signal = Signal
                for signal in tp_Signal:
                    temp_3 = feature_name
                    feature_name = feature_name + signal
                    for axis in Axis:
                        temp_4 = feature_name
                        feature_name = feature_name + axis
                        features_names.append(feature_name)
                        feature_name = temp_4
                    feature_name = temp_3
                feature_name = temp_2
            feature_name = temp_1
        feature_name = temp_0

    return features_names


def mag_f_features_names():
    # mean,std,median absolute deviation,max,min,interquartile range,sum of area,skewness, kurtosis,
    # mean_energy, freq of max, mean freq, #all bands energy
    Feature = ['Mean', 'Std', 'Mad', 'Max', 'Min', 'Iqr', 'Sma', 'Skewness', 'Kurtosis', 'MeanEnergy',
               'FreqOfMax', 'MeanFreq']
    Position = ['Hand', 'Chest']
    Sensor = ['Acc', 'Gyro']
    Signal = ['Grav', 'Body', 'Jerk']

    features_names = []
    feature_name = 'mag_f_'
    for feature in Feature:
        temp_0 = feature_name
        feature_name = feature_name + feature
        for position in Position:
            temp_1 = feature_name
            feature_name = feature_name + position
            for sensor in Sensor:
                temp_2 = feature_name
                feature_name = feature_name + sensor
                if sensor == 'Gyro':
                    tp_Signal = Signal[1:]
                else:
                    tp_Signal = Signal

                for signal in tp_Signal:
                    temp_3 = feature_name
                    feature_name = feature_name + signal
                    features_names.append(feature_name)
                    feature_name = temp_3
                feature_name = temp_2
            feature_name = temp_1
        feature_name = temp_0
    return features_names


def t_features_generate(t_data):
    """
        计算时域特征
        mean,std,median absolute deviation,max,min,interquartile range,sum of area,mean_energy,energy,
        pearsonr,skewness,kurtosis,4-order auto regression
    """
    t_mean = mean_axial(t_data)
    t_std = std_axial(t_data)
    t_mad = mad_axial(t_data)
    t_max = max_axial(t_data)
    t_min = min_axial(t_data)
    t_iqr = iqr_axial(t_data)
    t_sma = sma_axial(t_data)
    t_mean_energy = mean_energy_axial(t_data)
    t_energy = energy_axial(t_data)
    t_pearsonr = pearsonr_axial(t_data)
    t_skew = skewness_axial(t_data)
    t_kurt = kurtosis_axial(t_data)
    t_ar = arburg_axial(t_data)
    t_features_vector = t_mean + t_std + t_mad + t_max + t_min + t_iqr + t_sma + t_mean_energy + t_energy + t_pearsonr \
                        + t_skew + t_kurt + t_ar
    return t_features_vector


def mag_t_features_generate(mag_t_data):
    """
        计算时域幅度特征
        t_magnitude: mean,std,median absolute deviation,max,min,interquartile range,sum of area,mean_energy,
        energy,skewness,kurtosis,4-order auto regression
    """
    mag_t_mean = mean_axial(mag_t_data)
    mag_t_std = std_axial(mag_t_data)
    mag_t_mad = mad_axial(mag_t_data)
    mag_t_max = max_axial(mag_t_data)
    mag_t_min = min_axial(mag_t_data)
    mag_t_iqr = iqr_axial(mag_t_data)
    mag_t_sma = sma_axial(mag_t_data)
    mag_t_mean_energy = mean_energy_axial(mag_t_data)
    mag_t_energy = energy_axial(mag_t_data)
    mag_t_skew = skewness_axial(mag_t_data)
    mag_t_kurt = kurtosis_axial(mag_t_data)
    mag_t_ar = arburg_axial(mag_t_data)
    mag_t_features_vector = mag_t_mean + mag_t_std + mag_t_mad + mag_t_max + mag_t_min + mag_t_iqr + \
                            mag_t_sma + mag_t_mean_energy + mag_t_energy + mag_t_skew + mag_t_kurt + mag_t_ar
    return mag_t_features_vector


def f_features_generate(f_data):
    """
        计算频域特征
        mean,std,median absolute deviation,max,min,interquartile range,sum of area,skewness, kurtosis,
        mean_energy, freq of max, mean freq, all bands energy
    """
    f_mean = mean_axial(f_data)
    f_std = std_axial(f_data)
    f_mad = mad_axial(f_data)
    f_max = max_axial(f_data)
    f_min = min_axial(f_data)
    f_iqr = iqr_axial(f_data)
    f_sma = sma_axial(f_data)
    f_skew = skewness_axial(f_data)
    f_kurt = kurtosis_axial(f_data)
    f_mean_energy = mean_energy_axial(f_data)
    f_freq_max = max_freq_axial(f_data)
    f_mean_freq = f_mean_freq_axial(f_data)
    # f_bands_energy = fe.f_bands_energy_axial(f_data)
    features_vector = f_mean + f_std + f_mad + f_max + f_min + f_iqr + f_sma + f_skew + f_kurt + f_mean_energy + \
                      f_freq_max + f_mean_freq
    return features_vector


def mag_f_features_generate(mag_f_data):
    """
        计算频域幅度特征
        mean,std,median absolute deviation,max,min,interquartile range,sum of area,skewness, kurtosis, mean_energy,
        freq of max, mean freq
    """
    mag_f_mean = mean_axial(mag_f_data)
    mag_f_std = std_axial(mag_f_data)
    mag_f_mad = mad_axial(mag_f_data)
    mag_f_max = max_axial(mag_f_data)
    mag_f_min = min_axial(mag_f_data)
    mag_f_iqr = iqr_axial(mag_f_data)
    mag_f_sma = sma_axial(mag_f_data)
    mag_f_skew = skewness_axial(mag_f_data)
    mag_f_kurt = kurtosis_axial(mag_f_data)
    mag_f_mean_energy = mean_energy_axial(mag_f_data)
    mag_f_freq_max = max_freq_axial(mag_f_data)
    mag_f_mean_freq = f_mean_freq_axial(mag_f_data)
    # f_bands_energy = fe.f_bands_energy_axial(f_data)
    features_vector = mag_f_mean + mag_f_std + mag_f_mad + mag_f_max + mag_f_min + mag_f_iqr + mag_f_sma + mag_f_skew \
                      + mag_f_kurt + mag_f_mean_energy + mag_f_freq_max + mag_f_mean_freq
    features_names = mag_f_features_names()
    return features_vector

