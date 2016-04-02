from tensornet import TensorNet
from genericnet import GenericNet

class MNISTNet(GenericNet):

    input_channels = 1

    def __init__(self, arch, graph):
        self.arch = arch
        self.graph = graph
        self.generate_net()
        

    def _append_input(self):


    def _append_output(self):
        


    def generate_net(self):
        self.x = tf.placeholder('float', shape=[None, 28, 28, self.input_channels])
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        self.last_layer = self.x

        
