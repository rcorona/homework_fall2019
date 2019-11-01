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

def q3():
    
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/*PongNoFrameskip*')
    data_path = glob(os.path.join(data_path, 'events.out.tfevents*'))

    legend = []
    l = 600

    for hparam in data_path: 

        # Get return values. 
        avg_returns = np.asarray(parse_tf_events_file(hparam, 'Train_AverageReturn'), dtype=np.float32)[:l]

        # Plot result. 
        timesteps = np.asarray(parse_tf_events_file(hparam, 'Train_EnvstepsSoFar'), dtype=np.int32)[:l]
        plt.plot(timesteps, avg_returns)

        name = hparam.split('/')[9]
        
        if 'hparam1' in name: 
            name = '2'
        elif 'hparam2' in name: 
            name = '8'
        elif 'q1' in name:
            name = '32'
        elif 'hparam3' in name:
            name = '128'

        legend.append(name)

    # Save figure. 
    plt.legend(legend)
    plt.xlabel('Timesteps')
    plt.ylabel('Average Return')

    plt.title('Pong Q3 Batch Size Comparison')
    plt.savefig('q3.png')
    plt.close()

def q4(): 

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/ac_*CartPole*')
    data_path = glob(os.path.join(data_path, 'events.out.tfevents*'))

    legend = []

    for path in data_path: 

        # Get return values. 
        avg_return = np.asarray(parse_tf_events_file(path, 'Train_AverageReturn'), dtype=np.float32) 
        timesteps = np.asarray(parse_tf_events_file(path, 'Train_EnvstepsSoFar'), dtype=np.int32)
        
        plt.plot(timesteps, avg_return)

        ntu, ngsptu = path.split('ac')[1].split('_')[1:3]
        legend.append('ntu: {} ngsptu: {}'.format(ntu, ngsptu))

    # Save figure. 
    plt.legend(legend)
    plt.xlabel('Timesteps')
    plt.ylabel('Average Return')

    plt.title('Cart Pole Q4')
    plt.savefig('q4.png')
    plt.close()

def q5(): 

    ip_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/*InvertedPendulum*')
    ip_data_path = glob(os.path.join(ip_data_path, 'events.out.tfevents*'))[0]

    hc_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/*HalfCheetah*')
    hc_data_path = glob(os.path.join(hc_data_path, 'events.out.tfevents*'))[0]

    # Get return values. 
    ip_avg_returns = np.asarray(parse_tf_events_file(ip_data_path, 'Train_AverageReturn'), dtype=np.float32)
    hc_avg_returns = np.asarray(parse_tf_events_file(hc_data_path, 'Train_AverageReturn'), dtype=np.float32)

    ip_timesteps = np.asarray(parse_tf_events_file(ip_data_path, 'Train_EnvstepsSoFar'), dtype=np.int32)
    hc_timesteps = np.asarray(parse_tf_events_file(hc_data_path, 'Train_EnvstepsSoFar'), dtype=np.int32)

    # Plot results.
    plt.plot(ip_timesteps, ip_avg_returns)
    plt.xlabel('Timesteps')
    plt.ylabel('Average Return')

    plt.title('Inverted Pendulum Q5')
    plt.savefig('q5_ip.png')
    plt.close()

    plt.plot(hc_timesteps, hc_avg_returns)
    plt.xlabel('Timesteps')
    plt.ylabel('Average Return')

    plt.title('Half Cheetah Q5')
    plt.savefig('q5_hc.png')
    plt.close()

if __name__ == '__main__':
    
    args = parse_args()

    if args.problem == 2: 
        p2()
    elif args.problem == 3:
        p3()
    elif args.problem == 4: 
        p4()
