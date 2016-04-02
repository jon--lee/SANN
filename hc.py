
from arch import Arch
import random
from sann import SANN
import tensorflow as tf

class HillClimber(SANN):

    def __init__(self, initial_arch, T = 100):
        self.iterations = T
        self.current_arch = initial_arch
        self.max_hd = 4;

    def prob_function(self, e, eprime, T):
        if eprime < e:
            return 1
        else:
            return 0

    def iterate(self):
        curr_loss = self.current_arch.loss()
        
        neighbors = self.get_nearest_neighbors()
        arch_prime = self.choose_neighbor(neighbors)
        loss_prime = arch_prime.loss()

        prob = self.prob_function(curr_loss, loss_prime, 10)
        if prob > random.random():
            self.current_arch = arch_prime
            self.best = arch_prime
        

if __name__ == '__main__':
    init_arch = Arch.make_arch([.003, .3, .05, .05, 2, 2, [5, 3], [11, 5], [512, 256], tf.train.MomentumOptimizer])
    n = 1000
    s = HillClimber(init_arch, n)
    for i in range(n):
        s.iterate()
    print "final: " + str(s.current_arch.loss())
