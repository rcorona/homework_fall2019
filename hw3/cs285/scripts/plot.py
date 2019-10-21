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

def q1(): 

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/dqn_q1_PongNoFrameskip-v4_16-10-2019_05-22-18')
    data_path = glob(os.path.join(data_path, 'events.out.tfevents*'))[0]

    l=600

    # Get return values. 
    avg_return = np.asarray(parse_tf_events_file(data_path, 'Train_AverageReturn'), dtype=np.float32)[:l]
    best_return = np.asarray(parse_tf_events_file(data_path, 'Train_BestReturn'), dtype=np.float32)[:l]
    timesteps = np.asarray(parse_tf_events_file(data_path, 'Train_EnvstepsSoFar'), dtype=np.int32)[:l]

    # Plot result. 
    plt.plot(timesteps, avg_return)
    plt.plot(timesteps, best_return)

    # Save figure. 
    plt.legend(['Average Return', 'Best Return'])
    plt.xlabel('Timesteps')
    plt.ylabel('Return')

    plt.title('Pong Q1')
    plt.savefig('q1.png')
    plt.close()

def q2(): 

    dqn_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/dqn_q2_dqn*')
    dqn_data_path = glob(os.path.join(dqn_data_path, 'events.out.tfevents*'))

    ddqn_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/dqn_double*')
    ddqn_data_path = glob(os.path.join(ddqn_data_path, 'events.out.tfevents*'))

    dqn_avg_returns = []
    ddqn_avg_returns = []

    dqn_timesteps = []
    ddqn_timesteps = []

    for seed in dqn_data_path: 

        # Get return values. 
        dqn_avg_returns.append(np.asarray(parse_tf_events_file(seed, 'Train_AverageReturn'), dtype=np.float32))
        dqn_timesteps.append(np.asarray(parse_tf_events_file(seed, 'Train_EnvstepsSoFar'), dtype=np.int32)[:-1])

    for seed in ddqn_data_path: 

        # Get return values
        ddqn_avg_returns.append(np.asarray(parse_tf_events_file(seed, 'Train_AverageReturn'), dtype=np.float32))
        ddqn_timesteps.append(np.asarray(parse_tf_events_file(seed, 'Train_EnvstepsSoFar'), dtype=np.int32)[:-1])

    # Average over 3 seeds. 
    dqn_avg_returns = np.asarray(dqn_avg_returns)
    dqn_avg_returns = np.mean(dqn_avg_returns, 0)

    ddqn_avg_returns = np.asarray(ddqn_avg_returns)
    ddqn_avg_returns = np.mean(ddqn_avg_returns, 0)

    # Plot result. 
    plt.plot(dqn_timesteps[0], dqn_avg_returns)
    plt.plot(ddqn_timesteps[0], ddqn_avg_returns)

    # Save figure. 
    plt.legend(['DQN', 'DDQN'])
    plt.xlabel('Timesteps')
    plt.ylabel('Average Return')

    plt.title('Lunar Lander Q2')
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

    if args.problem == 1: 
        q1()
    elif args.problem == 2: 
        q2()
    elif args.problem == 3:
        q3()
    elif args.problem == 4: 
        q4()
    elif args.problem == 5: 
        q5()
