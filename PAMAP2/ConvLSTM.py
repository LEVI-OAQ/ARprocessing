import numpy as np
import _pickle as cp
import torch
from torch import nn
import torch.nn.functional as F
import warnings


warnings.filterwarnings('ignore')


def load_dataset(filename):
    with open(filename, 'rb') as f:
        data_x, data_y, data_group = cp.load(f)
    print(f" ..from file {filename}")
    print(f" ..reading instances: {data_x.shape}")
    data_x = data_x.astype(np.float32)
    data_y = data_y.astype(np.uint8)
    data_group = data_group.astype(np.uint8)
    return data_x, data_y, data_group


class HARModel(nn.Module):

    def __init__(self, n_channels = 18, window_len = 48, n_hidden = 16, n_layers = 1, n_filters = 64,
                 n_classes = 12, filter_size = 3, drop_prob = 0.2):
        super(HARModel, self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.filter_size = filter_size
        self.n_channels = n_channels
        self.window_len = window_len

        # Conv1d. Actually, filter_size is filter_size * in_channels
        self.conv1 = nn.Conv1d(n_channels, out_channels=n_filters, kernel_size=filter_size)
        self.conv2 = nn.Conv1d(n_filters, out_channels=n_filters, kernel_size=filter_size)
        self.conv3 = nn.Conv1d(n_filters, out_channels=n_filters, kernel_size=filter_size)
        self.conv4 = nn.Conv1d(n_filters, out_channels=n_filters, kernel_size=filter_size)

        self.lstm1 = nn.LSTM(n_filters, n_hidden, n_layers)
        self.lstm2 = nn.LSTM(n_hidden, n_hidden, n_layers)

        self.fc = nn.Linear(n_hidden, n_classes)

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, hidden, batch_size):
        # Conv1d  : input (N, C_in, L_in)  output(N, C_out, L_out)
        x = x.view(-1, self.n_channels, self.window_len)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # LSTM : input (seq_len, batch, input_size)
        x = x.view(40, -1, self.n_filters)
        x, hidden = self.lstm1(x, hidden)
        x, hidden = self.lstm2(x, hidden)  # ??? 需要把上一层的hidden传入么？

        # [8, 100, 128] => [800, 128] ??? 把 seq_len 和 batch_size 合在了一起
        x = x.contiguous().view(-1, self.n_hidden)
        x = self.dropout(x)
        x = self.fc(x)
        # LSTM, seq_len=8, only interest in the last result
        out = x.view(batch_size, -1, self.n_classes)[:, -1, :]

        return out, hidden

    def init_hidden(self, batch_size):
        """Initializes hidden state"""
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        train_on_gpu = torch.cuda.is_available()
        if train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        return hidden


class HARModelCrossSubjest(nn.Module):

    def __init__(self, n_channels=18, window_len=48, n_hidden=16, n_layers=1, n_filters=64,
                 n_classes=12, filter_size=3, drop_prob=0.5):
        super(HARModelCrossSubjest, self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.filter_size = filter_size
        self.n_channels = n_channels
        self.window_len = window_len

        # Conv1d. Actually, filter_size is filter_size * in_channels
        self.conv1 = nn.Conv1d(n_channels, out_channels=n_filters, kernel_size=filter_size)
        self.conv2 = nn.Conv1d(n_filters, out_channels=n_filters, kernel_size=filter_size)
        self.conv3 = nn.Conv1d(n_filters, out_channels=n_filters, kernel_size=filter_size)
        self.conv4 = nn.Conv1d(n_filters, out_channels=n_filters, kernel_size=filter_size)

        self.lstm1 = nn.LSTM(n_filters, n_hidden, n_layers)
        self.lstm2 = nn.LSTM(n_hidden, n_hidden, n_layers)

        self.fc = nn.Linear(n_hidden, n_classes)

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, hidden, batch_size):
        # Conv1d  : input (N, C_in, L_in)  output(N, C_out, L_out)
        x = x.view(-1, self.n_channels, self.window_len)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # LSTM : input (seq_len, batch, input_size)
        x = x.view(40, -1, self.n_filters)
        x, hidden = self.lstm1(x, hidden)
        x, hidden = self.lstm2(x, hidden)  # ??? 需要把上一层的hidden传入么？

        # [8, 100, 128] => [800, 128] ??? 把 seq_len 和 batch_size 合在了一起
        x = x.contiguous().view(-1, self.n_hidden)
        x = self.dropout(x)
        x = self.fc(x)
        # LSTM, seq_len=8, only interest in the last result
        out = x.view(batch_size, -1, self.n_classes)[:, -1, :]

        return out, hidden

    def init_hidden(self, batch_size):
        """Initializes hidden state"""
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        train_on_gpu = torch.cuda.is_available()
        if train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        return hidden


def init_weights(m):
    if type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif type(m) == nn.Conv1d or type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)


def iterate_minibatches(inputs, targets, batchsize, shuffle = True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(network, train_features, train_labels, test_features, test_labels, epochs = 100, batch_size = 100,
          lr = 0.02):
    opt = torch.optim.SGD(network.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # lr adjust, Multiplicative
    # lmbda = lambda epoch: 0.85
    # scheduler = torch.optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=lmbda)

    # lr adjust, ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        network.cuda()

    epo_train_loss = []
    epo_test_loss = []
    epo_accuracy = []
    confusion = np.zeros((12, 12), dtype=int)
    for e in range(epochs):
        # initialize hidden state
        h = network.init_hidden(batch_size)
        train_losses = []

        # if e > 30:
        #     lr = 0.015
        # if e > 60:
        #     lr = 0.001
        # if e > 70:
        #     lr = 0.0005
        # if e > 80:
        #     lr = 0.0001
        # if e > 85:
        #     lr = 0.00001
        # if e > 90:
        #     lr = 0.000001
        # if e > 95:
        #     lr = 0.0000001
        # if e > 100:
        #     lr = 0.00000001
        # adjust_learning_rate(opt, lr)

        # # 微调
        # if e > 20:
        #     lr = 0.015
        # if e > 40:
        #     lr = 0.0075
        # if e > 80:
        #     lr = 0.00375
        # if e > 90:
        #     lr = 0.001
        # if e > 120:
        #     lr = 0.0005
        # adjust_learning_rate(opt, lr)

        network.train()
        for batch in iterate_minibatches(train_features, train_labels, batch_size):
            x, y = batch

            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            # 每个batch训练得到的 h 都传入下一个batch，这里为了避免计算之前的梯度
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            opt.zero_grad()

            # get the output from the model
            output, h = network(inputs, h, batch_size)
            loss = criterion(output, targets.long())
            train_losses.append(loss.item())
            loss.backward()
            opt.step()

        # ----
        network.eval()
        val_h = network.init_hidden(batch_size)
        val_losses = []
        accuracy = 0
        with torch.no_grad():
            for batch in iterate_minibatches(test_features, test_labels, batch_size):
                x, y = batch

                inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

                val_h = tuple([each.data for each in val_h])

                if torch.cuda.is_available():
                    inputs, targets = inputs.cuda(), targets.cuda()

                output, val_h = network(inputs, val_h, batch_size)

                val_loss = criterion(output, targets.long())
                val_losses.append(val_loss.item())

                top_p, top_class = output.topk(1, dim=1)
                equals = top_class == targets.view(*top_class.shape).long()
                accuracy += torch.mean(equals.type(torch.FloatTensor))

                # confusion matrix
                for i in range(len(top_class)):
                    confusion[targets[i], top_class[i]] = confusion[targets[i], top_class[i]] + 1
                # plot_Matrix(confusion, 12, title=None, cmap=plt.cm.Blues)

        # lr adjust, ReduceLROnPlateau
        scheduler.step(np.mean(val_losses))

        # result exhibition
        epo_train_loss.append(np.mean(train_losses))
        epo_test_loss.append(np.mean(val_losses))
        epo_accuracy.append(accuracy / (len(test_features) // batch_size))
        confusion = confusion / np.sum(confusion, axis=1).reshape((-1, 1))
        confusion = confusion.round(2)
        print(confusion)
        print("Epoch: {}/{}...".format(e + 1, epochs),
              "Train Loss: {:.4f}...".format(np.mean(train_losses)),
              "Val Loss: {:.4f}...".format(np.mean(val_losses)),
              "Val Acc: {:.4f}...".format(accuracy / (len(test_features) // batch_size)))
    return epo_train_loss, epo_test_loss, epo_accuracy, confusion


def test(network, test_features, test_labels, batch_size = 100):

    if torch.cuda.is_available():
        network.cuda()
    criterion = nn.CrossEntropyLoss()
    confusion = np.zeros((12, 12), dtype=int)

    network.eval()
    test_h = network.init_hidden(batch_size)
    test_losses = []
    accuracy = 0
    with torch.no_grad():
        for batch in iterate_minibatches(test_features, test_labels, batch_size):
            x, y = batch

            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            test_h = tuple([each.data for each in test_h])

            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            output, test_h = network(inputs, test_h, batch_size)

            test_loss = criterion(output, targets.long())
            test_losses.append(test_loss.item())

            top_p, top_class = output.topk(1, dim=1)
            equals = top_class == targets.view(*top_class.shape).long()
            accuracy += torch.mean(equals.type(torch.FloatTensor))

            # confusion matrix
            for i in range(len(top_class)):
                confusion[targets[i], top_class[i]] = confusion[targets[i], top_class[i]] + 1

    confusion = confusion / np.sum(confusion, axis=1).reshape((-1, 1))
    confusion = confusion.round(2)
    accuracy = accuracy / (len(test_features) // batch_size)
    print(confusion)
    print("Test Loss: {:.4f}...".format(np.mean(test_losses)),
          "Test Acc: {:.4f}...".format(accuracy))

    return np.mean(test_losses), accuracy, confusion
