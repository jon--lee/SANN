"""
    Network architectures are specified

    [learning_rate (float), momentum (float), weight_init (float), bias_init (float),
    num_conv (int), num_fc (int), channels (list<int>), filters (list<int>), fc_dim (list<int>), optimizer (int) ]

"""

from tensornet import TensorNet
from controlnet import ControlNet
from mnistnet import MNISTNet
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
        self.T = 1
        self.dT = 1/float(T)
        self.max_hd = 4
        self.best = initial_arch


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
            print "hamming distance is: " + str(hd)
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
        print "T = " + str(self.T)
        curr_loss = self.current_arch.loss()
        print self.current_arch
        
        self.log_loss()
        self.log_acc()
        self.log_best_loss()
        self.log_best_acc()

        neighbors = self.get_nearest_neighbors()
        arch_prime = self.choose_neighbor(neighbors)
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
        path = "test_loss.log"
        f = open(path, 'a+')
        f.write(str(self.current_arch.loss()) + "\n")
        return 

    def log_acc(self):
        path = "test_acc.log"
        f = open(path, 'a+')
        f.write(str(self.current_arch.acc()) + "\n")
        return

    def log_best_acc(self):
        path = "best_acc.log"
        f = open(path, 'a+')
        f.write(str(self.best.acc()) + "\n")
        return 

    def log_best_loss(self):
        path = "best_loss.log"
        f = open(path, 'a+')
        f.write(str(self.best.loss()) + "\n")
        return        
        return
    


    
    
if __name__ == '__main__':
    good_params = {'convs': 2, 'channels': [4, 3], 'weight_init': 0.5, 'fcs': 3, 'lr': 0.006, 'bias_init': 0.1, 'filters': [11, 7], 'optimizer': tf.train.AdagradOptimizer, 'mo': 0.9, 'fc_dim': [128, 256, 64]}
    #init_arch = Arch.make_arch([1e-4, .3, .05, .05, 2, 1, [32, 64], [5, 5], [512], tf.train.AdamOptimizer])
    init_arch = Arch(good_params)
    g = tf.Graph()
    #data = AMTData('data/train.txt', 'data/test.txt', channels=3)
    data = MNISTData() 
    with g.as_default():
        net = MNISTNet(init_arch, g)
        loss, acc, _ = net.optimize(500, data, batch_size=100)
        print "Loss: " + str(loss)
        print "Accuracy: " + str(acc)
    
    
    
    
    #n = 1000
    #s = SANN(init_arch, n)
    #for i in range(n):
    #    s.iterate()
    """g = tf.Graph()
    with g.as_default():
        net = ControlNet(init_arch, g);
    """
