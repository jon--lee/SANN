from genericnet import GenericNet
from tensornet import TensorNet
import tensorflow as tf
"""
    All control nets nets will have at least one conv leading and four output control nodes

    include method to initialize network with given parameters
    net essentially uses add conv and add fc to to do so and finally append output



"""

class ControlNet(GenericNet):

    input_channels = 3

    def _append_input(self):
        self.x = tf.placeholder('float', shape=[None, 250, 250, self.input_channels])
        self.y_ = tf.placeholder('float', shape=[None, 4])
        self.last_layer = self.x
        
    
    def _append_output(self):
        num_nodes = abs(TensorNet.reduce_shape(self.last_layer.get_shape()))
        flattened = tf.reshape(self.last_layer, [-1, num_nodes]) 

        self.w_fc_out = self.weight_variable([num_nodes, 4])
        self.b_fc_out = self.bias_variable([4])

        self.y_out = tf.tanh(tf.matmul(flattened, self.w_fc_out) + self.b_fc_out)
        self.loss = tf.reduce_mean(.5*tf.square(self.y_out - self.y_))
        
        wrapper = GenericNet.opt_wrapper(self.arch.optimizer)
        self.train_step = wrapper(self.arch.lr, self.arch.mo)
        self.train = self.train_step.minimize(self.loss)


    
