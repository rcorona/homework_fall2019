import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from glob import glob
import os
import tensorflow as tf
import numpy as np

# From Piazza post @85. 
def parse_tf_events_file(filename):
    eval_returns = []
    for e in tf.train.summary_iterator(filename):
        for v in e.summary.value:
            if v.tag == 'Eval_AverageReturn':
                eval_returns.append(v.simple_value)

    return eval_returns

def p7(): 

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')
    results = glob(os.path.join(data_path, '*HalfCheetah*'))
    
    # Will hold legend labels. 
    legend = []

    for r_dir in results: 
        
        # First get experiment results from tf events file. 
        tf_path = glob(os.path.join(r_dir, 'events.out.tfevents*'))[0]
        avg_return = np.asarray(parse_tf_events_file(tf_path), dtype=np.float32)

        # Extract experiment name for legend. 
        name = '_'.join(os.path.basename(r_dir).split('_')[2:4])
        legend.append(name)

        # Plot result. 
        plt.plot(np.arange(len(avg_return)), avg_return)

    # Make legend. 
    plt.legend(legend)
    plt.savefig('test_fig.png')

if __name__ == '__main__':
    p7()
