import torch
import torch.nn as nn
import numpy as np
import config


cfg = config.CONFIG()
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)


class SparseDropout(torch.nn.Module):
    def __init__(self, dprob=0.5):
        super(SparseDropout, self).__init__()
        # dprob is ratio of dropout
        # convert to keep probability
        self.kprob=1-dprob

    def forward(self, x):
        mask=((torch.rand(x._values().size())+(self.kprob)).floor()).type(torch.bool)
        rc=x._indices()[:,mask]
        val=x._values()[mask]*(1.0/self.kprob)
        return torch.sparse.FloatTensor(rc, val)


class GraphConvolution(nn.Module):
    def __init__( self, input_dim,
                        output_dim,
                        support,
                        act_func = None,
                        featureless = False,
                        dropout_rate = 0.,
                        bias=False,
                        layer=1):
        super(GraphConvolution, self).__init__()
        self.support = support
        self.featureless = featureless

        setattr(self, 'W', nn.Parameter(torch.randn(input_dim, output_dim)))

        if bias:
            self.b = nn.Parameter(torch.zeros(1, output_dim))

        self.act_func = act_func
        self.layer = layer
        if self.layer == 1:
            self.dropout = SparseDropout(dropout_rate)
        elif self.layer == 2:
            self.dropout = nn.Dropout(dropout_rate)

        self.embedding = None


    def forward(self, x):
        x = self.dropout(x)

        if self.featureless:
            pre_sup = getattr(self, 'W')
        else:
            if self.layer == 1:
                pre_sup = torch.sparse.mm(x, getattr(self, 'W'))
            elif self.layer == 2:
                pre_sup = x.mm(getattr(self, 'W'))

        # out = self.support.mm(pre_sup)
        out = torch.sparse.mm(self.support, pre_sup)

        if self.act_func is not None:
            out = self.act_func(out)
        self.embedding = out
        return out


class GCN(nn.Module):
    def __init__( self, input_dim,
                        support,
                        dropout_rate=0.,
                        num_classes=10,
                        train_size=None,
                        test_size=None):
        super(GCN, self).__init__()

        self.test_size = test_size
        self.train_size = train_size
        
        # GraphConvolution
        self.support = support

        self.layer1 = GraphConvolution(input_dim, 200, self.support, act_func=nn.ReLU(), featureless=False, dropout_rate=dropout_rate, layer=1)
        self.layer2 = GraphConvolution(200, num_classes, self.support, dropout_rate=dropout_rate, layer=2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out
