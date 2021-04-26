"""
    re-organize the ConvLSTM model, but failed with unknown reason
"""
# import numpy as np
# import _pickle as cp
# import matplotlib.pyplot as plt
# import pandas as pd
# import sklearn.metrics as metrics
# import torch
# from torch import nn
# import torch.nn.functional as F
# import warnings
# from PAMAP2.sliding_window import sliding_window
#
# warnings.filterwarnings('ignore')
#
#
# def plot_Matrix(cm, n_classes, title = None, cmap = plt.cm.Blues):
#     plt.rc('font', family='Times New Roman', size='8')  # 设置字体样式、大小
#
#     # 按行进行归一化
#     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     print("Normalized confusion matrix")
#     str_cm = cm.astype(np.str).tolist()
#     for row in str_cm:
#         print('\t'.join(row))
#     # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             if int(cm[i, j] * 100 + 0.5) == 0:
#                 cm[i, j] = 0
#
#     fig, ax = plt.subplots()
#     im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#     # ax.figure.colorbar(im, ax=ax) # 侧边的颜色条带
#
#     ax.set(xticks=np.arange(cm.shape[1]),
#            yticks=np.arange(cm.shape[0]),
#            xticklabels=n_classes, yticklabels=n_classes,
#            title=title,
#            ylabel='Actual',
#            xlabel='Predicted')
#
#     # 通过绘制格网，模拟每个单元格的边框
#     ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
#     ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
#     ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
#     ax.tick_params(which="minor", bottom=False, left=False)
#
#     # 将x轴上的lables旋转45度
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")
#
#     # 标注百分比信息
#     fmt = 'd'
#     thresh = cm.max() / 2.
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             if int(cm[i, j] * 100 + 0.5) > 0:
#                 ax.text(j, i, format(int(cm[i, j] * 100 + 0.5), fmt) + '%',
#                         ha="center", va="center",
#                         color="white" if cm[i, j] > thresh else "black")
#     fig.tight_layout()
#     plt.savefig('cm.jpg', dpi=300)
#     plt.show()
#
#
# def load_dataset(filename):
#     with open(filename, 'rb') as f:
#         data = cp.load(f)
#
#     x_train, y_train = data[0]
#     x_test, y_test = data[1]
#
#     print(" ..from file {}".format(filename))
#     print(" ..reading instances: train {0}, test {1}".format(x_train.shape, x_test.shape))
#
#     x_train = x_train.astype(np.float32)
#     x_test = x_test.astype(np.float32)
#
#     # The targets are casted to int8 for GPU compatibility.
#     y_train = y_train.astype(np.uint8)
#     y_test = y_test.astype(np.uint8)
#     return x_train, y_train, x_test, y_test
#
#
# def opp_sliding_window(data_x, data_y, ws, ss):
#     data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
#     data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])  # 取最后一个值，作为标签
#     return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)
#
#
# class HARModel(nn.Module):
#
#     def __init__(self, n_hidden = 16, n_layers = 1, n_filters = 64,
#                  n_classes = 12, filter_size = 3, drop_prob = 0.2):
#         super(HARModel, self).__init__()
#         self.drop_prob = drop_prob
#         self.n_layers = n_layers
#         self.n_hidden = n_hidden
#         self.n_filters = n_filters
#         self.n_classes = n_classes
#         self.filter_size = filter_size
#
#         self.cnn_stack = nn.Sequential(
#             nn.Conv1d(NB_SENSOR_CHANNELS, out_channels=n_filters, kernel_size=filter_size),
#             nn.ReLU(),
#             nn.Conv1d(n_filters, out_channels=n_filters, kernel_size=filter_size),
#             nn.ReLU(),
#             nn.Dropout(drop_prob),
#             nn.Conv1d(n_filters, out_channels=n_filters, kernel_size=filter_size),
#             nn.ReLU(),
#             nn.Conv1d(n_filters, out_channels=n_filters, kernel_size=filter_size),
#             nn.ReLU()
#         )
#         self.lstm_stack = nn.Sequential(
#             nn.LSTM(n_filters, n_hidden, n_layers),
#             nn.LSTM(n_hidden, n_hidden, n_layers),
#         )
#
#         self.fc_stack = nn.Sequential(
#             nn.Dropout(drop_prob),
#             nn.Linear(n_hidden, n_classes)
#         )
#
#     def forward(self, x, hidden, batch_size):
#         # Conv1d  : input (N, C_in, L_in)  output(N, C_out, L_out)
#         x = x.view(-1, NB_SENSOR_CHANNELS, SLIDING_WINDOW_LENGTH)
#         x = self.cnn_stack(x)
#
#         # LSTM : input (seq_len, batch, input_size)
#         x = x.view(40, -1, self.n_filters)
#         x, hidden = self.lstm_stack(x, hidden)
#
#         # [8, 100, 128] => [800, 128] ??? 把 seq_len 和 batch_size 合在了一起
#         x = x.contiguous().view(-1, self.n_hidden)
#         x = self.fc_stack(x)
#         # LSTM, seq_len=8, only interest in the last result
#         out = x.view(batch_size, -1, self.n_classes)[:, -1, :]
#         return out, hidden
#
#     def init_hidden(self, batch_size):
#         """Initializes hidden state"""
#         # Create two new tensors with sizes n_layers x batch_size x n_hidden,
#         # initialized to zero, for hidden state and cell state of LSTM
#         weight = next(self.parameters()).data
#
#         if train_on_gpu:
#             hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
#                       weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
#         else:
#             hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
#                       weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
#         return hidden
#
#
# def init_weights(m):
#     if type(m) == nn.LSTM:
#         for name, param in m.named_parameters():
#             if 'weight_ih' in name:
#                 torch.nn.init.orthogonal_(param.data)
#             elif 'weight_hh' in name:
#                 torch.nn.init.orthogonal_(param.data)
#             elif 'bias' in name:
#                 param.data.fill_(0)
#     elif type(m) == nn.Conv1d or type(m) == nn.Linear:
#         torch.nn.init.orthogonal_(m.weight)
#         m.bias.data.fill_(0)
#
#
# def iterate_minibatches(inputs, targets, batchsize, shuffle = True):
#     assert len(inputs) == len(targets)
#     if shuffle:
#         indices = np.arange(len(inputs))
#         np.random.shuffle(indices)
#     for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
#         if shuffle:
#             excerpt = indices[start_idx:start_idx + batchsize]
#         else:
#             excerpt = slice(start_idx, start_idx + batchsize)
#         yield inputs[excerpt], targets[excerpt]
#
#
# def train(network, epochs = 100, batch_size = 100, lr = 0.05):
#     opt = torch.optim.SGD(network.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
#     criterion = nn.CrossEntropyLoss()
#     if train_on_gpu:
#         network.cuda()
#
#     epo_train_loss = []
#     epo_test_loss = []
#     epo_accuracy = []
#     for e in range(epochs):
#         confusion = np.zeros((12, 12))
#         # initialize hidden state
#         h = network.init_hidden(batch_size)
#         train_losses = []
#
#         network.train()
#         for batch in iterate_minibatches(X_train, Y_train, batch_size):
#             x, y = batch
#
#             inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
#
#             if train_on_gpu:
#                 inputs, targets = inputs.cuda(), targets.cuda()
#
#             # 每个batch训练得到的 h 都传入下一个batch，这里为了避免计算之前的梯度
#             # Creating new variables for the hidden state, otherwise
#             # we'd backprop through the entire training history
#             h = tuple([each.data for each in h])
#
#             # zero accumulated gradients
#             opt.zero_grad()
#
#             # get the output from the model
#             output, h = network(inputs, h, batch_size)
#             loss = criterion(output, targets.long())
#             train_losses.append(loss.item())
#             loss.backward()
#             opt.step()
#
#         # ----
#         network.eval()
#         val_h = network.init_hidden(batch_size)
#         val_losses = []
#         accuracy = 0
#         f1score = 0
#         with torch.no_grad():
#             for batch in iterate_minibatches(X_test, Y_test, batch_size):
#                 x, y = batch
#
#                 inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
#
#                 val_h = tuple([each.data for each in val_h])
#
#                 if train_on_gpu:
#                     inputs, targets = inputs.cuda(), targets.cuda()
#
#                 output, val_h = network(inputs, val_h, batch_size)
#
#                 val_loss = criterion(output, targets.long())
#                 val_losses.append(val_loss.item())
#
#                 top_p, top_class = output.topk(1, dim=1)
#                 equals = top_class == targets.view(*top_class.shape).long()
#                 accuracy += torch.mean(equals.type(torch.FloatTensor))
#
#                 # confusion
#                 for i in range(len(top_class)):
#                     confusion[targets[i], top_class[i]] = confusion[targets[i], top_class[i]] + 1
#                 # plot_Matrix(confusion, 12, title=None, cmap=plt.cm.Blues)
#         scheduler.step(np.mean(val_losses))
#
#         epo_train_loss.append(np.mean(train_losses))
#         epo_test_loss.append(np.mean(val_losses))
#         epo_accuracy.append(accuracy / (len(X_test) // batch_size))
#
#         print("Epoch: {}/{}...".format(e + 1, epochs),
#               "Train Loss: {:.4f}...".format(np.mean(train_losses)),
#               "Val Loss: {:.4f}...".format(np.mean(val_losses)),
#               "Val Acc: {:.4f}...".format(accuracy / (len(X_test) // batch_size)))
#
#     fig, ax = plt.subplots()
#     ax.plot(np.arange(e + 1), epo_train_loss, label='train_loss')
#     ax.plot(np.arange(e + 1), epo_test_loss, label='epo_test_loss')
#     ax.plot(np.arange(e + 1), epo_accuracy, label='epo_accuracy')
#     ax.set_xlabel('epochs')
#     ax.set_ylabel('loss/accuracy')
#     ax.set_title("result")
#     ax.legend()
#     plt.show()
#
#
# NB_SENSOR_CHANNELS = 18
# SLIDING_WINDOW_LENGTH = 48
#
# print("Loading data...")
# X_train, Y_train, X_test, Y_test = load_dataset('../PAMAP2/Protocol/ahrs/preprocessed_data_toChestFrame')
# print("..train data.shape: ", X_train.shape)
# print("..test data.shape: ", X_test.shape)
#
# net = HARModel()
# net.apply(init_weights)
# # check if GPU is available
# train_on_gpu = torch.cuda.is_available()
# if train_on_gpu:
#     print('Training on GPU!')
# else:
#     print('No GPU available, training on CPU;')
#
# print("training...")
# train(net)
#
#