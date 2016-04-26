"""
    Network architectures are specified

    [learning_rate (float), momentum (float), weight_init (float), bias_init (float),
    num_conv (int), num_fc (int), channels (list<int>), filters (list<int>), fc_dim (list<int>), optimizer (int) ]

"""

from nets.tensornet import TensorNet
from nets.controlnet import ControlNet
from nets.mnistnet import MNISTNet
from inputdata import MNISTData
from arch import Arch
from inputdata import AMTData
import itertools
import numpy as np
import copy
import random
import tensorflow as tf

class SANN():
    
    def __init__(self, initial_arch, T = 100):
        self.current_arch = initial_arch
        self.iterations = T
        self.T = 1.0
        self.dT = self.T/float(T)
        self.max_hd = 2
        self.best = initial_arch
        
        self.best_acc_path = "logs/best_acc_sann.log"
        self.best_loss_path = "logs/best_loss_sann.log"
        self.test_loss_path = "logs/test_loss_sann.log"
        self.test_acc_path = "logs/test_acc_sann.log"

    @staticmethod
    def prob_function(e, eprime, T):
        if eprime < e:
            return 1
        elif eprime > 1e40:
            return 0
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
            print "Hamming distance is: " + str(hd)
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
        print "[ Optimization Step ] T = " + str(self.T)
        print "Current arch: " + str(self.current_arch)        
        curr_loss = self.current_arch.loss()
        
        self.log_loss()
        self.log_acc()
        self.log_best_loss()
        self.log_best_acc()

        neighbors = self.get_nearest_neighbors()
        arch_prime = self.choose_neighbor(neighbors)
        print "Prime now testing: " + str(arch_prime)
        loss_prime = arch_prime.loss()

        prob = self.prob_function(curr_loss, loss_prime, self.T)
        if prob > random.random():
            if prob < 1.0:
                print "JUMPING ANYWAY"
            self.current_arch = arch_prime
        self.T = max(self.T - self.dT, 1e-9)
        if self.current_arch.loss() < self.best.loss():
            self.best = self.current_arch
        print "\n\n\n\n"

    def run(self):
        self.best = self.current_arch
        for _ in range(self.iterations):
            self.iterate()


    @staticmethod
    def hamming_dist(arch1, arch2):
        s = 0
        for el1, el2 in zip(arch1, arch2):
            if not el1 == el2:
                s += 1
        return s


    def log_loss(self):
        f = open(self.test_loss_path, 'a+')
        f.write(str(self.current_arch.loss()) + "\n")
        return 

    def log_acc(self):
        f = open(self.test_acc_path, 'a+')
        f.write(str(self.current_arch.acc()) + "\n")
        return

    def log_best_acc(self):
        f = open(self.best_acc_path, 'a+')
        f.write(str(self.best.acc()) + "\n")
        return 

    def log_best_loss(self):
        f = open(self.best_loss_path, 'a+')
        f.write(str(self.best.loss()) + "\n")
        return
    


    
    
if __name__ == '__main__':
    params1 = {'convs': 2, 'channels': [32, 64], 'weight_init': 0.1, 'fcs': 1, 'lr': 0.0001, 'bias_init': 0.5, 'filters': [5, 5], 'optimizer': tf.train.AdamOptimizer, 'mo': 0.09, 'fc_dim': [512]}
    params2 = {'convs': 2, 'channels': [32, 64], 'weight_init': 0.1, 'fcs': 1, 'lr': 0.0001, 'bias_init': 0.5, 'filters': [5, 5], 'optimizer': tf.train.AdamOptimizer, 'mo': 0.09, 'fc_dim': [512]}
    arch1 = Arch.make_arch(params1)
    arch2 = Arch.make_arch(params2)
    print SANN.hamming_dist(arch1, arch2)
    
    
