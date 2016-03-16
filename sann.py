"""
    Network architectures are specified

    [learning_rate (float), momentum (float), weight_init (float), bias_init (float),
    num_conv (int), num_fc (int), channels (list<int>), filters (list<int>), fc_dim (list<int>), optimizer (int) ]

"""

from tensornet import TensorNet
from controlnet import ControlNet
from arch import Arch
from inputdata import AMTData
import itertools
import numpy as np
import copy
import random
import tensorflow as tf

class SANN():
    
    def __init__(self, initial_arch, T = 100, dT = 1):
        self.current_arch = initial_arch
        self.T = 1
        self.dT = float(dT)/float(T)
        self.max_hd = 4

    @staticmethod
    def prob_function(e, eprime, T):
        if eprime < e:
            return 1
        else:
            return np.exp(-(eprime - e)/T)

    def choose_neighbor(self, neighbors):
        return random.choice(neighbors)


    def get_nearest_neighbors(self):
        neighbor_dict = self._rand_param()
        neighbor = Arch(neighbor_dict)
        hd = SANN.hamming_dist(self.current_arch, neighbor)
        if hd == 0 or hd > self.max_hd:
            return self.get_nearest_neighbors()
        else:
            return [neighbor]


    def _rand_param(self):
        curr_dict = self.current_arch.todict()
        neighbor_dict = copy.copy(curr_dict)
        for key, options in Arch.possible_values.iteritems():
            if key == 'channels' or key == 'filters' or key == 'fc_dim':
                continue
            if random.random() > .8:    # perturb state
                choice = random.choice(Arch.possible_values[key])
                
                if key == 'convs':
                    neighbor_dict['channels'] = SANN._rand_subset(Arch.possible_values['channels'], choice)
                    neighbor_dict['filters'] = SANN._rand_subset(Arch.possible_values['filters'], choice)
                elif key == 'fcs':
                    neighbor_dict['fc_dim'] = SANN._rand_subset(Arch.possible_values['fc_dim'], choice)
                neighbor_dict[key] = choice
            else:
                neighbor_dict[key] = curr_dict[key]

        return neighbor_dict

    @staticmethod
    def _rand_subset(st, n):
        st = list(st)
        subset = []
        for i in range(n):
            choice = random.choice(st)
            subset.append(choice)
            st.remove(choice)
        return subset


    def iterate(self):
        curr_loss = self.current_arch.loss()
        neighbors = self.get_nearest_neighbors()
        arch_prime = self.choose_neighbor(neighbors)
        loss_prime = arch_prime.loss()
    
        print loss_prime 

        prob = self.prob_function(curr_loss, loss_prime, self.T)
        if prob > random.random():
            print "choosing prime, prob: " + str(prob)            
            self.current_arch = arch_prime
            self.T = max(self.T - self.dT, 0.0)
        else:
            a = 1
            print "prob: " + str(prob)
        
    @staticmethod
    def hamming_dist(arch1, arch2):
        s = 0
        for el1, el2 in zip(arch1, arch2):
            if not el1 == el2:
                s += 1
        return s

    
    
if __name__ == '__main__':
    init_arch = Arch.make_arch([.003, .3, .05, .05, 2, 2, [5, 3], [11, 5], [512, 256], tf.train.MomentumOptimizer])
    s = SANN(init_arch)
    print "initial: " + str(s.current_arch.loss())
    for i in range(100):
        s.iterate()
    print "final: " + str(s.current_arch.loss())
    """g = tf.Graph()
    with g.as_default():
        net = ControlNet(init_arch, g);
    """
