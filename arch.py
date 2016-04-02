"""
    [learning_rate (float), momentum (float), weight_init (float), bias_init (float),
    num_conv (int), num_fc (int), channels (list<int>), filters (list<int>), fc_dim (list<int>), optimizer (int) ]
"""
import tensorflow as tf
from controlnet import ControlNet

class Arch():

    keys = ['lr', 'mo', 'weight_init', 'bias_init', 'convs',
            'fcs', 'channels', 'filters', 'fc_dim', 'optimizer']

    possible_values = {
        "lr": sorted([.0009, .003, .006, .009, .03, .06, .09]),
        "mo": sorted([.09, .5, .9]),
        "weight_init": sorted([.005, .05, .5]),
        "bias_init": sorted([0.01, 0.1, .5]),
        "convs": sorted([1, 2, 3, 4]),   
        "fcs": sorted([0, 1, 2, 3]), 
        # reverse order matters for these since they go biggest->smallest and depend on num of fc an conv
        "channels": sorted([2, 3, 4, 5, 6], reverse=True),
        "filters": sorted([3, 5, 7, 11], reverse=True),
        "fc_dim": sorted([32, 64, 128, 256, 512], reverse=True),
        "optimizer": [tf.train.MomentumOptimizer, tf.train.AdagradOptimizer, tf.train.GradientDescentOptimizer]
    }

    def __init__(self, params):
        self.lr = params['lr']
        self.mo = params['mo']
        self.weight_init = params['weight_init']
        self.bias_init = params['bias_init']
        self.convs = params['convs']
        self.fcs = params['fcs']
        self.channels = params['channels']
        self.filters = params['filters']
        self.fc_dim = params['fc_dim']
        self.optimizer = params['optimizer']
        self._loss = -1


    def __iter__(self):
        return iter(self.tolist())

    def tolist(self):
        return [ self.lr, self.mo, self.weight_init, self.bias_init, self.convs,
            self.fcs, self.channels, self.filters, self.fc_dim, self.optimizer]

    def todict(self):
        return {
            'lr': self.lr,
            'mo': self.mo,
            'weight_init': self.weight_init,
            'bias_init': self.bias_init,
            'convs': self.convs,
            'fcs': self.fcs,
            'channels': self.channels,
            'filters': self.filters,
            'fc_dim': self.fc_dim,
            'optimizer': self.optimizer
        }

    def __str__(self):
        return str(self.todict())

    def _repr__(self):
        return str(self)

    def loss(self):
        if self._loss > -1:
            return self._loss
        return self._compute_loss()

    def _compute_loss(self):
        # TODO: actually (train and) compute loss by constructing graph with net
        g = tf.Graph()
        with g.as_default():
            net = ControlNet(self.current_arch, g)
            net.train()
            
        self._loss = (50*self.lr)**3 + (50*self.mo)**2 + (10*self.weight_init)**2 + (10*self.bias_init)**4 + (self.convs)**5 + (self.fcs)**2
        return self._loss
    
    @staticmethod
    def make_arch(lst):
        params = { key: val for key, val in zip(Arch.keys, lst) }
        return Arch(params)
        
