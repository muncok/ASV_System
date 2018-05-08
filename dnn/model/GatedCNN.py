
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import SerializableModule

class GatedCNN(SerializableModule):
    '''
        In : (N, sentence_len)
        Out: (N, sentence_len, embd_size)
    '''
    def __init__(self,
                 seq_len,
                 embd_size,
                 n_layers,
                 kernel,
                 out_chs,
                 res_block_count,
                 ans_size):
        super(GatedCNN, self).__init__()
        self.res_block_count = res_block_count
        self.limit = n_layers

        # self.embedding = nn.Embedding(vocab_size, embd_size)

        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...
        time_pad= kernel[0]//2
        self.conv_0 = nn.Conv2d(1, out_chs, kernel, padding=(time_pad, 0))
        self.b_0 = nn.Parameter(torch.randn(1, out_chs, 1, 1))
        self.conv_gate_0 = nn.Conv2d(1, out_chs, kernel, padding=(time_pad, 0))
        self.c_0 = nn.Parameter(torch.randn(1, out_chs, 1, 1))

        self.conv = nn.ModuleList([nn.Conv2d(out_chs, out_chs, (kernel[0], 1),
                                             padding=(time_pad, 0)) for _ in range(n_layers)])
        self.conv_gate = nn.ModuleList([nn.Conv2d(out_chs, out_chs, (kernel[0], 1),
                                                  padding=(time_pad, 0)) for _ in range(n_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.randn(1, out_chs, 1, 1))
                                   for _ in range(n_layers)])
        self.c = nn.ParameterList([nn.Parameter(torch.randn(1, out_chs, 1, 1))
                                   for _ in range(n_layers)])

        self.output = nn.Linear(out_chs*seq_len, ans_size)

        # for param in self.parameters():
            # if len(param.size()) > 1:
                # nn.init.kaiming_normal(param)

        self.feat_size = out_chs

    def embed(self, x):
        # x: (N, seq_len)

        # Embedding
        # seq_len = x.size(1)
        # x = self.embedding(x) # (bs, seq_len, embd_size)

        # Conv2d
        #    Input : (bs, Cin,  Hin,  Win )
        #    Output: (bs, Cout, Hout, Wout)
        A = self.conv_0(x)      # (bs, Cout, seq_len, 1)
        A += self.b_0.repeat(1, 1, A.size(2), 1)
        B = self.conv_gate_0(x) # (bs, Cout, seq_len, 1)
        B += self.c_0.repeat(1, 1, B.size(2), 1)
        h = A * F.sigmoid(B)    # (bs, Cout, seq_len, 1)
        res_input = h # TODO this is h1 not h0

        for i, conv, conv_gate in zip(range(self.limit), self.conv, self.conv_gate):
            A = conv(h)
            A += self.b[i].repeat(1, 1, A.size(2), 1)
            B = conv_gate(h)
            B += self.c[i].repeat(1, 1, B.size(2), 1)
            h = A * F.sigmoid(B) # (bs, Cout, seq_len, 1)
            if i % self.res_block_count == 0: # size of each residual block
                h += res_input
                res_input = h
        # h = torch.mean(h, 2) # mean over seq axis
        # out = h.squeeze_()
        return h

    def forward(self, x):
        bs = x.size(0) # batch size
        h = self.embed(x)
        h = h.view(bs, -1) # (bs, Cout*seq_len)
        out = self.output(h) # (bs, ans_size)
        return out

