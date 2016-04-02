from tensornet import TensorNet
import tensorflow as tf
"""
    All control nets nets will have at least one conv leading and four output control nodes

    include method to initialize network with given parameters
    net essentially uses add conv and add fc to to do so and finally append output



"""

class GenericNet(TensorNet):
    
    input_channels = 0
    
    def __init__(self, arch, graph):
        self.arch = arch
        self.graph = graph 
        self._generate_net()


    def _append_input(self):
        raise NotImplementedError

    def _generate_net(self):
        raise NotImplementedError
        
        
    def _add_conv(self, filter_size, depth):
        last_layer_depth = self.last_layer.get_shape().as_list()[-1]
        w_conv = self.weight_variable([filter_size, filter_size, last_layer_depth, depth])
        b_conv = self.bias_variable([depth])
        h_conv = tf.nn.relu(self.conv2d(self.last_layer, w_conv) + b_conv)
        self.last_layer = h_conv


    def _add_fc(self, num_fc_nodes):
        num_nodes = abs(TensorNet.reduce_shape(self.last_layer.get_shape()))
        flattened = tf.reshape(self.last_layer, [-1, num_nodes])
        
        w_fc = self.weight_variable([num_nodes, num_fc_nodes])
        b_fc = self.bias_variable([num_fc_nodes])

        h_fc = tf.nn.relu(tf.matmul(flattened, w_fc) + b_fc)
        self.last_layer = h_fc


    def _append_output(self):
        raise NotImplementedError


    @staticmethod
    def opt_wrapper(optimizer):
        if optimizer == tf.train.MomentumOptimizer:
            return optimizer
        else:
            return lambda x, y: optimizer(x)

    def weight_variable(self, shape):
        return TensorNet.weight_variable(self, shape, self.arch.weight_init)


    def bias_variable(self, shape):
        return TensorNet.bias_variable(self, shape, self.arch.bias_init)
        



