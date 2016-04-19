from sann import SANN
from hc import HillClimber
from gridsearch import GridSearch
from inputdata import MNISTData
from arch import Arch
import itertools
import random
import tensorflow as tf

# note! don't try this, it takes too long (billions of permutations, likely hours to run)
# it's cool as hell though
def generate_permutations():
    vals = Arch.possible_values
    p_set = []
    count = 0
    for convs, fcs, mo, lr, weight_init, bias_init, opt in itertools.product(vals['convs'], vals['fcs'], vals['mo'], vals['lr'], vals['weight_init'], vals['bias_init'], vals['optimizer']):
    #for lr, mo, convs, fcs, weight_init, bias_init, opt in itertools.product(vals['lr'], vals['mo'], vals['convs'], vals['fcs'], vals['weight_init'],
    #        vals['bias_init'], vals['optimizer']):
        filter_options_set = [ vals['filters'] for _ in range(convs) ]
        filter_options = [ r for r in itertools.product(*filter_options_set) ]
        channel_options_set = [vals['channels'] for _ in range(convs) ]
        channel_options = [ r for r in itertools.product(*channel_options_set) ]
        fcdim_options_set = [ vals['fc_dim'] for _ in range(fcs) ]
        fcdim_options = [ r for r in itertools.product(*fcdim_options_set) ]
        
        for filters in filter_options:
            for channels in channel_options:
                for fcdim in fcdim_options:
                    p = {'lr': lr, 'mo': mo, 'weight_init': weight_init, 'bias_init': bias_init,
                            'convs': convs, 'fcs': fcs, 'channels': channels, 'filters': filters,
                            'fc_dim': fcdim, 'optimizer': opt
                            }
                    p = {'convs': convs, 'fcs': fcs, 'filters': filters, 'channels': channels, 'fc_dim': fcdim}
                    p_set.append(p)
    print len(p_set)
    return p_set


def get_random_arch():
    params = {}
    for key in Arch.keys:
        options = Arch.possible_values[key]
        if key == 'channels' or key == 'filters' or key == 'fc_dim':
            continue
        choice = random.choice(Arch.possible_values[key])
        if key == 'convs':
            params['channels'] = SANN._rand_subset(Arch.possible_values['channels'], choice)
            params['filters'] = SANN._rand_subset(Arch.possible_values['filters'], choice)
        elif key == 'fcs':
            params['fc_dim'] = SANN._rand_subset(Arch.possible_values['fc_dim'], choice)            
        params[key] = choice
    return params


def permute_archs():
    p_sets = generate_permutations()
    archs = []
    return archs
    #for p in p_sets:
    #    arch = Arch(p)
    #    archs.append(arch)
    #return archs




if __name__ == '__main__':
    #archs = permute_archs()
    sann_losses = []
    hc_losses = []
    gs_losses = []
    #print get_random_arch()
    for _ in range(1):
        #init_arch = random.choice(archs)
        #init_params = get_random_arch()
        init_params = {
                'convs': 2, 'channels': [3, 5], 
                'weight_init': 0.5, 'fcs': 3, 'lr': 0.006,
                'bias_init': 0.1, 'filters': [3, 11], 
                'optimizer': tf.train.AdagradOptimizer, 
                'mo': 0.5, 'fc_dim': [128, 256, 64]
        }
        """init_params = {
                'convs': 2, 'channels': [32, 64], 
                'weight_init': 0.1, 'fcs': 1, 'lr': 0.0009,
                'bias_init': 0.1, 'filters': [5, 5], 
                'optimizer': tf.train.AdamOptimizer, 
                'mo': 0.5, 'fc_dim': [512]
        }"""
        init_arch = Arch(init_params)
        print "initial loss: " + str(init_arch.loss())

        s = SANN(init_arch, T = 50)
        #hc = HillClimber(init_arch, 80)
        
        s.run()
        #hc.run()

        print "\n\nFinal SANN Loss: " + str(s.best.loss())
        print "Final SANN Accuracy: " + str(s.best.acc())
        print "Final SANN: " + str(s.best)
        #print "final hc: " + str(hc.best.loss())


    
