from tensornet import TensorNet
from genericnet import GenericNet
import tensorflow as tf
import datetime
class MNISTNet(GenericNet):

    input_channels = 1

    def _append_input(self):
        self.x = tf.placeholder('float', shape=[None, 28, 28, self.input_channels])
        self.y_ = tf.placeholder('float', shape=[None, 10])
        self.last_layer = self.x


    def _append_output(self):
        num_nodes = abs(TensorNet.reduce_shape(self.last_layer.get_shape()))

        flattened = tf.reshape(self.last_layer, [-1, num_nodes]) 

        self.w_fc_out = self.weight_variable([num_nodes, 10])
        self.b_fc_out = self.bias_variable([10])

        self.y_out = tf.nn.softmax(tf.matmul(flattened, self.w_fc_out) + self.b_fc_out)
        print self.y_out.get_shape()
        self.loss = -tf.reduce_sum(self.y_*tf.log(self.y_out))
        
        wrapper = GenericNet.opt_wrapper(self.arch.optimizer)
        self.train_step = wrapper(self.arch.lr, self.arch.mo)
        self.train = self.train_step.minimize(self.loss)

        self.correct_prediction = tf.equal(tf.argmax(self.y_out,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        


    def optimize(self, iterations, data, path=None, batch_size=100, test_print=40, save=True):
        if path:
            sess = self.load(var_path=path)
        else:
            print "Initializing new variables..."
            NUM_CORES = 1
            sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES, intra_op_parallelism_threads=NUM_CORES))
            sess.run(tf.initialize_all_variables())
            
        self.log_path = self.dir + 'train.log'
        #logging.basicConfig(filename=log_path, level=logging.DEBUG)
        failed = False
        try:
            with sess.as_default():
                for i in range(iterations):
                    batch = data.next_train_batch(batch_size)
                    ims, labels = batch

                    feed_dict = { self.x: ims, self.y_: labels }
                    if i % 20 == 0:
                        batch_loss = self.loss.eval(feed_dict=feed_dict) / batch_size
                        self.log("[ Iteration " + str(i) + " ] Training loss: " + str(batch_loss))
                    if i % test_print == 0:
                        test_batch = data.next_validation_batch()
                        test_ims, test_labels = test_batch
                        test_dict = { self.x: test_ims, self.y_: test_labels }
                        test_loss = self.loss.eval(feed_dict=test_dict) / len(test_ims)
                        self.log("[ Iteration " + str(i) + " ] Validation loss: " + str(test_loss ))
                    self.train.run(feed_dict=feed_dict)
                
                test_batch = data.next_test_batch()
                test_ims, test_labels = test_batch
                test_dict = { self.x: test_ims, self.y_: test_labels }
                test_loss = self.loss.eval(feed_dict=test_dict) / len(test_ims)
                test_accuracy = self.accuracy.eval(feed_dict=test_dict)
                
        except:
            "Ran into an exception, giving up!"
            failed = True
            pass
        
        
        if not failed:
            if path:
                dir, old_name = os.path.split(path)
                dir = dir + '/'
            else:
                dir = 'models/'
            new_name = self.name + "_" + datetime.datetime.now().strftime("%m-%d-%Y_%Hh%Mm%Ss") + ".ckpt"
            if save:
                save_path = self.save(sess, save_path='models/' + new_name)
            else:
                save_path = None
            sess.close()
            self.log( "Optimization done." )
            return test_loss, test_accuracy, save_path
        else:
            return 1e50, 0.0, None
        

