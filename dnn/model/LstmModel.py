import torch
import torch.nn as nn
from torch.autograd import Variable

from .model import SerializableModule

class SpeechLSTMModel(SerializableModule):
    def __init__(self, config):
        super().__init__()
        target_size = config["n_labels"]
        no_cuda = config['no_cuda']
        self.input_dim = config["in_dim"]
        self.hidden_dim = config["h_dim"]
        self.n_layers = config["n_layers"]
        self.batch_size = config["batch_size"]

        self.lstm = nn.LSTM(input_size=self.input_dim , hidden_size=self.hidden_dim, num_layers=self.n_layers)
        self.proj = nn.Linear(self.hidden_dim, target_size)
        self.hidden = self.init_hidden(no_cuda=no_cuda)
        self.feat_size = self.hidden_dim

    def init_hidden(self, no_cuda=False):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        hidden_init = Variable(nn.init.uniform(torch.zeros(self.n_layers, self.batch_size, self.hidden_dim),
                                               a=-0.02, b=0.02))
        cell_init = Variable(nn.init.uniform(torch.zeros(self.n_layers, self.batch_size, self.hidden_dim),
                                                a=-0.02, b=0.02))
        if not no_cuda:
            hidden_init = hidden_init.cuda()
            cell_init = cell_init.cuda()
        return (hidden_init, cell_init)

    def embed(self, x):
        # input: (sequence, batch, features)
        x.transpose_(0,1)

        # hidden = self.hidden
        # x = x.contiguous()
        # for i in x:
        # # Step through the sequence one element at a time.
        # # after each step, hidden contains the hidden state.
        # i = i.view(1, self.batch_size, -1)
        # out, hidden = self.lstm(i, hidden)

        lstm_out, hidden = self.lstm(x, self.hidden)
        last_out = lstm_out[-1]
        return last_out

    def forward(self, x):
        last_out = self.embed(x)
        proj_out = self.proj(last_out)
        return proj_out
