from PAMAP2.preprocess_data import data_generate


DIR_PATH = '../PAMAP2/Protocol/ahrs/'
TARGET = 'withoutTransfer.data'
data_generate(DIR_PATH, TARGET, window_width=48, stride_len=24)
