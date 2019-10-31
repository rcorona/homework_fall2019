import numpy as np

from .base_policy import BasePolicy


class MPCPolicy(BasePolicy):

    def __init__(self,
        sess,
        env,
        ac_dim,
        dyn_models,
        horizon,
        N,
        **kwargs):
        super().__init__(**kwargs)

        # init vars
        self.sess = sess
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

    def sample_action_sequences(self, num_sequences, horizon):
        # TODO(Q1) uniformly sample trajectories and return an array of
        # dimensions (num_sequences, horizon, self.ac_dim)
        return np.random.uniform(self.low, self.high, [num_sequences, horizon, self.ac_dim]) 

    def get_action(self, obs):

        if self.data_statistics is None:
            # print("WARNING: performing random actions.")
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        #sample random actions (Nxhorizon)
        candidate_action_sequences = self.sample_action_sequences(num_sequences=self.N, horizon=self.horizon)

        # a list you can use for storing the predicted reward for each candidate sequence
        predicted_rewards_per_ens = []

        for model in self.dyn_models:
            # TODO(Q2)

            # for each candidate action sequence, predict a sequence of
            # states for each dynamics model in your ensemble

            # once you have a sequence of predicted states from each model in your
            # ensemble, calculate the reward for each sequence using self.env.get_reward(predicted_obs)
            predicted_rewards = []
            obs_pl = np.tile(np.reshape(obs, [1, -1]), (len(candidate_action_sequences), 1))

            for i in range(candidate_action_sequences.shape[1]):
                acs = candidate_action_sequences[:,i]
                obs_pl = model.get_prediction(obs_pl, acs, self.data_statistics)
                
                rewards, _ = self.env.get_reward(obs_pl, acs)
                predicted_rewards.append(rewards)

            predicted_rewards = np.asarray(predicted_rewards)
            predicted_rewards = np.sum(predicted_rewards, 0)
            predicted_rewards_per_ens.append(predicted_rewards)

        predicted_rewards_per_ens = np.asarray(predicted_rewards_per_ens)

        # calculate mean_across_ensembles(predicted rewards).
        # the matrix dimensions should change as follows: [ens,N] --> N
        
        # TODO(Q2)
        predicted_rewards_per_ens = np.mean(predicted_rewards_per_ens, axis=0) 

        # pick the action sequence and return the 1st element of that sequence
        best_index = np.argmax(predicted_rewards_per_ens)  #TODO(Q2)
        best_action_sequence = candidate_action_sequences[best_index] #TODO(Q2)
        action_to_take = best_action_sequence[0] # TODO(Q2)
        return action_to_take[None] # the None is for matching expected dimensions
