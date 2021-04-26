from PAMAP2.ConvLSTM import train, test, load_dataset, HARModel, init_weights, HARModelCrossSubjest
from sklearn.model_selection import StratifiedShuffleSplit, GroupKFold
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')


print("Loading data...")
DIR_PATH = '../PAMAP2/Protocol/ahrs/'
FILE = 'withoutTransfer'
data_x, data_y, data_group = load_dataset(DIR_PATH + FILE + '.data')
n_splits = 9
cv_cnt = 0
epochs = 150
n_classes = 12

cross_train_loss = []
cross_test_loss = []
cross_accuracy = []
train_test_subjects = []
cross_valid_accuracy = []
LABELS = ['lie', 'sit', 'std', 'wlk', 'run', 'cyc', 'nlk', 'ups',
                   'dns', 'vac', 'iro', 'jmp']

# # train & test split
# tts = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
# # train & valid split
# tvs = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=42)


print('cross-subject kFold test...')
# train & test split
tts = GroupKFold(n_splits=n_splits)
# train & valid split
tvs = GroupKFold(n_splits=n_splits-1)


for train_index, test_index in tts.split(data_x, data_y, data_group):

    X_train, X_test = data_x[train_index], data_x[test_index]
    Y_train, Y_test = data_y[train_index], data_y[test_index]
    G_train = data_group[train_index]

    cv_cnt = cv_cnt + 1
    v_cnt = 0

    train_subjects = str(set(data_group[train_index]))
    test_subjects = str(set(data_group[test_index]))
    train_test_subjects.append(train_subjects + '//' + test_subjects)

    print(f"{cv_cnt} training...")
    print(f'train subjects: {train_subjects}')
    print(f'test subjects: {test_subjects}')
    
    net = HARModelCrossSubjest()
    net.apply(init_weights)

    for train_index_s, valid_index_s in tvs.split(X_train, Y_train, G_train):

        X_train_s, X_valid_s = X_train[train_index_s], X_train[valid_index_s]
        Y_train_s, Y_valid_s = Y_train[train_index_s], Y_train[valid_index_s]

        print(f'train subjects: {set(G_train[train_index_s])}')
        print(f'valid subjects: {set(G_train[valid_index_s])}')

        train_loss, valid_loss, valid_accuracy, _ = train(net, X_train_s, Y_train_s, X_valid_s, Y_valid_s, epochs=epochs)

        cross_train_loss.append(np.mean(train_loss[-5:]))
        cross_valid_accuracy.append(np.mean(valid_accuracy[-5:]))

        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(np.arange(epochs), train_loss, label='train_loss')
        ax1.plot(np.arange(epochs), valid_loss, label='valid_loss')
        ax1.set_ylabel('loss')
        ax1.legend()
        ax2.plot(np.arange(epochs), valid_accuracy)
        ax2.set_ylabel('accuracy')
        ax2.set_xlabel('epoch')

        v_cnt = v_cnt + 1
        fig.suptitle(f'{cv_cnt} train subjects:{set(G_train[train_index_s])} valid subjects:{set(G_train[valid_index_s])}')
        if not os.path.exists(DIR_PATH + FILE):
            os.mkdir(DIR_PATH + FILE)
        plt.savefig(DIR_PATH + FILE + f'/result{cv_cnt}-{v_cnt}.jpg')

        break

    print(f"{cv_cnt} testing...")
    test_loss, accuracy, confusion = test(net, X_test, Y_test)
    plt.matshow(confusion)
    plt.xticks(np.arange(n_classes), LABELS)
    plt.yticks(np.arange(n_classes), LABELS)
    plt.savefig(DIR_PATH + FILE + f'/confusion{cv_cnt}.jpg')

    cross_test_loss.append(test_loss)
    cross_accuracy.append(accuracy)


with open(DIR_PATH+FILE+'/result.txt', 'w') as f:
    for i in range(n_splits):
        f.write(f'{train_test_subjects[i]}\t{cross_train_loss[i]}\t{cross_test_loss[i]}\t{cross_accuracy[i]}\n')

