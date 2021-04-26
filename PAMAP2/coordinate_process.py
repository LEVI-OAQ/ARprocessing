"""
    , linear interpolation
    , calculate the quarternion at each sample
    , save data
"""

from ahrs.filters import EKF, Mahony
import numpy as np
import os
import pandas as pd
DIR_PATH = 'D:/pyProjects/ARprocessing/PAMAP2/Protocol/'

# – 0 timestamp (s)
# – 1 activityID (see II.2. for the mapping to the activities)
# – 2 heart rate (bpm)
# - IMU hand
#   3 temperature
#   4,5,6 acc
#   7,8,9 xxx
#   10,11,12 gyro
#   13,14,15 mag
#   16,17,18,19  quarternion
# - IMU chest
#   20 temperature
#   21,22,23 acc
#   24,25,26 xxx
#   27,28,29 gyro
#   30,31,32 mag
#   33,34,35,36  quarternion
# - IMU ankle
#   37 temperature
#   38,39,40 acc
#   41,42,43 xxx
#   44,45,46 gyro
#   47,48,49 mag
#   50,51,52,53  quarternion

ACC_INX = [4, 5, 6]
GYRO_INX = [10, 11, 12]
MAG_INX = [13, 14, 15]
QUAR_INX = [16, 17, 18, 19]
INX_OFFSET = 17

TARGET = os.path.join(DIR_PATH, 'ahrs/mahony/')
if os.path.exists(TARGET):
    os.rmdir(TARGET)
os.mkdir(TARGET)
files = os.listdir(DIR_PATH)
for file in files:
    if file.endswith('.dat'):
        data = np.loadtxt(os.path.join(DIR_PATH, file))
        print(f'{file} is processing...')
        for r in range(3):
            acc_inx = [i + r * INX_OFFSET for i in ACC_INX]
            gyro_inx = [i + r * INX_OFFSET for i in GYRO_INX]
            mag_inx = [i + r * INX_OFFSET for i in MAG_INX]
            quar_inx = [i + r * INX_OFFSET for i in QUAR_INX]
            # liner interpolate
            data[:, acc_inx] = np.array([pd.Series(i).interpolate() for i in data[:, acc_inx].T]).T
            data[:, gyro_inx] = np.array([pd.Series(i).interpolate() for i in data[:, gyro_inx].T]).T
            data[:, mag_inx] = np.array([pd.Series(i).interpolate() for i in data[:, mag_inx].T]).T
            # calculate quarternion
            # ekf = EKF(acc=data[:, acc_inx], gyr=data[:, gyro_inx], mag=data[:, mag_inx], frequency=100.0)
            # data[:, quar_inx] = ekf.Q
            mahony = Mahony(acc=data[:, acc_inx], gyr=data[:, gyro_inx], mag=data[:, mag_inx], frequency=100.0)
            data[:, quar_inx] = mahony.Q
        np.savetxt(os.path.join(TARGET, file), data, fmt='%.10f', delimiter='\t')
