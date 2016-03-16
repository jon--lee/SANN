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
import random
import tensorflow as tf

class SANN():
    
    def __init__(self, initial_arch, T = 25, dT = 1):
        self.current_arch = initial_arch
        self.T = 1
        self.dT = float(dT)/float(T)

    @staticmethod
    def prob_function(e, eprime, T):
        if eprime < e:
            return 1
        else:
            return np.exp(-(eprime - e)/T)

    def choose_neighbor(self, neighbors):
        return random.choice(neighbors)


    def get_nearest_neighbors(self):
        neighbor_dict = _rand_param()
        
        if SANN.hamming_dist(self.current_arch, neighbor) == 0:
            print "hamming distance conflict"
            return get_nearest_neighbors()
        else:
            print "selected neighbor is " + str(SANN.hamming_dist(self.current_arch, neighbor)) + " hamming distance"
            return [neighbor]


    def _rand_param(self):
        neighbor_dict = {}
        curr_dict = self.current_arch.todict()
        for key, options in Arch.possible_values.iteritems():
            if key == 'channels' or key == 'filters' or key == 'fc_dim':
                continue
            if random.random() > .8:    # perturb state
                choice = random.choice(Arch.possible_values[key])
                
                if key == 'convs':
                    neighbor_dict['channels'] = _rand_subset(Arch.possible_values['channels'], choice)
                    neighbor_dict['filters'] = _rand_subset(Arch.possible_values['filters'], choice)
                elif key == 'fcs':
                    neighbor_dict['fc_dim'] = _rand_subset(Arch.possible_values['fc_dim'], choice)
                neighbor_dict[key] = choice
            else:
                neighbor_dict[key] = curr_dict[key]
        return neighbor_dict


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
        neighbors = get_nearest_neighbors()
        arch_prime = choose_neighbor(neighbors)
        loss_prime = arch_prime.loss()

        if prob_function(curr_loss, loss_prime, self.T) > random.random():
            self.current_arch = arch_prime
        
        
    @staticmethod
    def hamming_dist(arch1, arch2):
        s = 0
        for el1, el2 in zip(arch1, arch2):
            if not el1 == el2:
                s += 1
        return s

    
    
if __name__ == '__main__':
    init_arch = Arch.make_arch([.003, .3, .05, .05, 2, 2, [5, 3], [11, 5], [512, 256], tf.train.MomentumOptimizer])
    
    """g = tf.Graph()
    with g.as_default():
        net = ControlNet(init_arch, g);
    """
