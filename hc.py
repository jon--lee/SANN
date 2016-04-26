
from arch import Arch
import random
from sann import SANN
import tensorflow as tf

class HillClimber(SANN):

    def __init__(self, initial_arch, T = 100):
        SANN.__init__(self, initial_arch, T)

        self.test_loss_path = "logs/test_loss_hc.log"
        self.test_acc_path = "logs/test_acc_hc.log"
        self.best_acc_path = "logs/best_acc_hc.log"
        self.best_loss_path = "logs/best_loss_hc.log"


    def prob_function(self, e, eprime, T):
        if eprime < e:
            return 1
        else:
            return 0

        

if __name__ == '__main__':
    init_arch = Arch.make_arch([.003, .3, .05, .05, 2, 2, [5, 3], [11, 5], [512, 256], tf.train.MomentumOptimizer])
    n = 1000
    s = HillClimber(init_arch, n)
    for i in range(n):
        s.iterate()
    print "final: " + str(s.current_arch.loss())
