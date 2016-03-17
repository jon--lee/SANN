from arch import Arch
from sann import SANN
import tensorflow as tf


class GridSearch(SANN):

    def __init__(self):
        self.current_arch = initial_arch
        

    def prob_function(e, eprime, T):
        if eprime < e:
            return 1
        else:
            return 0

    def get_nearest_neighbors(self):
        return [1]




    def iterate():
        curr_loss = self.current_arch.loss()
        neighbors = self.get_nearest_neighbors()
        arch_prime = self.choose_neighbor(neighbors)
        loss_prime = arch_prime.loss()

        prob = self.prob_function(curr_loss, loss_prime, 10)
        if prob > random.random():
            self.current_arch = arch_prime


if __name__ == '__main__':
    init_arch = Arch.make_arch([.003, .3, .05, .05, 2, 2, [5, 3], [11, 5], [512, 256], tf.train.MomentumOptimizer])
    n = 1000
    s = GridSearch(init_arch, n)
    for i in range(n):
        s.iterate()
    print "final: " + str(s.current_arch.loss())

