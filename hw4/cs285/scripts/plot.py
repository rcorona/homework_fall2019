import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from glob import glob
import os
import tensorflow as tf
import numpy as np
import argparse

# From Piazza post @85. 
def parse_tf_events_file(filename, tag):
    eval_returns = []
    for e in tf.train.summary_iterator(filename):
        for v in e.summary.value:

            #print(v.tag)

            if v.tag == tag:
                eval_returns.append(v.simple_value)

    return eval_returns

def parse_args(): 

    parser = argparse.ArgumentParser()

    parser.add_argument('-file_prefix', type=str, help='Prefix for file paths to read in for results.')
    parser.add_argument('-problem', type=int, help='Problem number to plot')

    return parser.parse_args()

def p2(): 

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/*obstacles_singleiteration*31-10*')
    data_path = glob(os.path.join(data_path, 'events.out.tfevents*'))[0]

    train_avg = parse_tf_events_file(data_path, 'Train_AverageReturn')
    eval_avg = parse_tf_events_file(data_path, 'Eval_AverageReturn')

    # Plot result. 
    plt.plot(np.arange(len(train_avg)), train_avg, marker='o')
    plt.plot(np.arange(len(eval_avg)), eval_avg, marker='x')

    # Save figure. 
    plt.legend(['Train', 'Eval'])
    plt.xlabel('Iterations')
    plt.ylabel('Average Return')

    plt.title('Obstacles Single Iteration Q2')
    plt.savefig('q2.png')
    plt.close()

def p3():

    # Obstacles. 
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/*obstacles_obstacles*31-10*35-21*')
    data_path = glob(os.path.join(data_path, 'events.out.tfevents*'))[0]

    eval_avg = parse_tf_events_file(data_path, 'Eval_AverageReturn')

    # Plot result. 
    plt.plot(np.arange(len(eval_avg)), eval_avg)

    # Save figure. 
    plt.xlabel('Iterations')
    plt.ylabel('Average Return')
    plt.ylim([-50.0, -20.0])

    plt.title('Obstacles Q3')
    plt.savefig('q3_obstacles.png')
    plt.close()

    # Reacher. 
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/*reacher_reacher*31-10*')
    data_path = glob(os.path.join(data_path, 'events.out.tfevents*'))[0]

    eval_avg = parse_tf_events_file(data_path, 'Eval_AverageReturn')

    # Plot result. 
    plt.plot(np.arange(len(eval_avg)), eval_avg)

    # Save figure. 
    plt.xlabel('Iterations')
    plt.ylabel('Average Return')

    plt.title('Reacher Q3')
    plt.savefig('q3_reacher.png')
    plt.close()

    # Half-Cheetah. 
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/*cheetah_cheetah*31-10*')
    data_path = glob(os.path.join(data_path, 'events.out.tfevents*'))[0]

    eval_avg = parse_tf_events_file(data_path, 'Eval_AverageReturn')

    # Plot result. 
    plt.plot(np.arange(len(eval_avg)), eval_avg)

    # Save figure. 
    plt.xlabel('Iterations')
    plt.ylabel('Average Return')

    plt.title('Cheetah Q3')
    plt.savefig('q3_cheetah.png')
    plt.close()

def p4(): 

    # Ensemble size. 
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/mb_q5_reacher_ensemble*31-10-2019*')
    data_path = glob(os.path.join(data_path, 'events.out.tfevents*'))

    legend = []

    for path in data_path: 

        eval_avg = parse_tf_events_file(path, 'Eval_AverageReturn')

        # Plot result. 
        plt.plot(np.arange(len(eval_avg)), eval_avg)

        # Extract name for legend. 
        name = path.split('/')[-2].split('_')[3].split('ensemble')[-1]
        
        legend.append(name)

    # Save figure. 
    plt.xlabel('Iterations')
    plt.ylabel('Average Return')
    plt.legend(legend)

    plt.title('Reacher Q4 Ensemble Size Comparison')
    plt.savefig('q4_ensemble.png')
    plt.close()

if __name__ == '__main__':
    
    args = parse_args()

    if args.problem == 2: 
        p2()
    elif args.problem == 3:
        p3()
    elif args.problem == 4: 
        p4()
