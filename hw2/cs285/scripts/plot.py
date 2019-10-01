import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from glob import glob
import os
import tensorflow as tf
import numpy as np
import argparse

# From Piazza post @85. 
def parse_tf_events_file(filename):
    eval_returns = []
    for e in tf.train.summary_iterator(filename):
        for v in e.summary.value:
            if v.tag == 'Eval_AverageReturn':
                eval_returns.append(v.simple_value)

    return eval_returns

def parse_args(): 

    parser = argparse.ArgumentParser()

    parser.add_argument('-file_prefix', type=str, help='Prefix for file paths to read in for results.')
    parser.add_argument('-problem', type=int, help='Problem number to plot')

    return parser.parse_args()

def p3(): 

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')
    sb_results = glob(os.path.join(data_path, 'pg_sb*')) 
    lb_results = glob(os.path.join(data_path, 'pg_lb*'))

    for results in (sb_results, lb_results): 

        legend = []

        for r_dir in results: 
            
            # First get experiment results from tf events file. 
            tf_path = glob(os.path.join(r_dir, 'events.out.tfevents*'))[0]
            avg_return = np.asarray(parse_tf_events_file(tf_path), dtype=np.float32)

            # Extract experiment name for legend. 
            name = '_'.join(os.path.basename(r_dir).split('_Cart')[0].split('_')[2:])
            legend.append(name)

            # Plot result. 
            plt.plot(np.arange(len(avg_return)), avg_return)

        # Save figure. 
        plt.legend(legend)
        plt.xlabel('Iteration')
        plt.ylabel('Avg. Return')

        if results == sb_results:
            f_name = 'sb'
        else: 
            f_name = 'lb'

        plt.title('Cart Pole {}'.format(f_name.upper()))
        plt.savefig('p3_{}.png'.format(f_name))
        plt.close()

def p4(): 

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')
    results = glob(os.path.join(data_path, 'pg_ip*v2_28*'))
    legend = []

    for r_dir in results: 

        # First get experiment results from tf events file. 
        tf_path = glob(os.path.join(r_dir, 'events.out.tfevents*'))[0]
        avg_return = np.asarray(parse_tf_events_file(tf_path), dtype=np.float32)

        # Extract experiment name for legend. 
        batch, lr = os.path.basename(r_dir).split('_')[2:4]

        # Plot result. 
        plt.plot(np.arange(len(avg_return)), avg_return)

    # Save figure. 
    plt.title('Inverted Pendulum -- Batch: {}, LR: {}'.format(batch, lr))
    plt.xlabel('Iteration')
    plt.ylabel('Avg. Return')
    plt.savefig('p4.png')
    plt.close()

def p6(): 

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')
    results = glob(os.path.join(data_path, 'pg_ll*'))

    for r_dir in results: 

        # First get experiment results from tf events file. 
        tf_path = glob(os.path.join(r_dir, 'events.out.tfevents*'))[0]
        avg_return = np.asarray(parse_tf_events_file(tf_path), dtype=np.float32)

        # Plot result. 
        plt.plot(np.arange(len(avg_return)), avg_return)

    # Save figure. 
    plt.title('Lunar Lander')
    plt.xlabel('Iteration')
    plt.ylabel('Avg. Return')
    plt.savefig('p6.png')
    plt.close()

def p7(): 

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')
    search_results = glob(os.path.join(data_path, 'pg_hc_*nnbaseline*'))
    final_results = glob(os.path.join(data_path, 'pg_hc_b30000_r0.02*'))

    for results in (search_results, final_results): 

        legend = []
        
        for r_dir in results: 
            
            # First get experiment results from tf events file. 
            tf_path = glob(os.path.join(r_dir, 'events.out.tfevents*'))[0]
            avg_return = np.asarray(parse_tf_events_file(tf_path), dtype=np.float32)

            # Extract experiment name for legend. 
            if results == search_results: 
                name = '_'.join(os.path.basename(r_dir).split('_')[2:4])
            elif results == final_results: 
                name = os.path.basename(r_dir).split('Half')[0].split('_')[-1]

                if name == '1':
                    name = 'no_rtg_no_baseline'
                elif name == '2':
                    name = 'rtg'
                elif name == '3':
                    name = 'baseline'
                elif name == '4':
                    name = 'rtg_baseline'

            name = name.strip('_')
            legend.append(name)

            # Plot result. 
            plt.plot(np.arange(len(avg_return)), avg_return)

        # Make legend. 
        plt.xlabel('Iteration')
        plt.ylabel('Avg. Return')
        plt.legend(legend)

        if results == search_results: 
            plt.title('Half-Cheetah Grid Search') 
            plt.savefig('p7_search.png')
        elif results == final_results:
            plt.title('Half-Cheetah Batch: {}, LR: {}'.format(30000, 0.02))
            plt.savefig('p7_results.png')

        plt.close()

if __name__ == '__main__':
    
    args = parse_args()

    if args.problem == 3: 
        p3()
    elif args.problem == 4: 
        p4()
    elif args.problem == 6:
        p6()
    elif args.problem == 7: 
        p7()
