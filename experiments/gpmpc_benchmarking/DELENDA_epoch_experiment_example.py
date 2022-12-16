import os

import yaml
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import munch

from safe_control_gym.utils.registration import make
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.utils import mkdirs, set_dir_from_config
from safe_control_gym.experiments.epoch_experiment import EpochExp

# To set relative pathing of experiment imports.
import sys
import os.path as path
from gpmpc_plotting_utils import gather_training_samples


def gather_training_samples(trajs_data, episode_i, num_samples, rand_generator=None):
    """
    # todo: Needs to de updated once data structures are modified to handle epochs.
    Data structure is now
        trajs_data.<var>[<episode #>].
        e.g. epsiode 4 states is traj_data.state[3]
    :param all_runs:
    :param n_episodes:
    :param num_samples:
    :param rand_generator:
    :return:
    """
    #num_samples_per_episode = int(num_samples/n_episodes)
    num_samples_per_episode = int(num_samples)
    x_seq_int = []
    x_next_seq_int = []
    actions_int = []
    n = trajs_data['action'][episode_i].shape[0]
    if num_samples_per_episode < n:
        if rand_generator is not None:
            rand_inds_int = rand_generator.choice(n-1, num_samples_per_episode, replace=False)
        else:
            rand_inds_int = np.arange(num_samples_per_episode)
    else:
        rand_inds_int = np.arange(n-1)
    next_inds_int = rand_inds_int + 1
    x_seq_int.append(trajs_data.obs[episode_i][rand_inds_int, :])
    actions_int.append(trajs_data.action[episode_i][rand_inds_int, :])
    x_next_seq_int.append(trajs_data.obs[episode_i][next_inds_int, :])
    x_seq_int = np.vstack(x_seq_int)
    actions_int = np.vstack(actions_int)
    x_next_seq_int = np.vstack(x_next_seq_int)

    return x_seq_int, actions_int, x_next_seq_int

def make_traking_plot(runs, traj, dir, plot_one_test=True, impossible=True):
    num_epochs = len(runs['state'])
    if plot_one_test:
        num_tests = 1
    else:
        num_tests = len(runs['state'][0])
    plt.figure()
    for test in range(0, num_tests):
        plt.plot(runs['state'][0][test][:, 0], runs['state'][0][test][:, 2], label=f'Linear MPC {test}')
        traj_lin = np.vstack((runs['state'][0][test][:, 0], runs['state'][0][test][:, 2])).T
        np.savetxt(os.path.join(dir, f'traj_lin_mpc_{test}.csv'), traj_lin, delimiter=',')
    for epoch in range(1, num_epochs):
        for test in range(num_tests):
            traj1 = np.vstack((runs['state'][epoch][test][:, 0], runs['state'][epoch][test][:, 2])).T
            np.savetxt(os.path.join(dir, f'traj_epoch_{epoch}_test_{test}.csv'), traj1, delimiter=',')
            plt.plot(runs['state'][epoch][test][:, 0], runs['state'][epoch][test][:, 2], label=f'GP-MPC e{epoch} t{test}')
    plt.plot(traj[:,0], traj[:,2], 'k',label='Reference')
    if impossible:
        plt.plot([-0.55,-0.55],[-0.1, 1.05], 'r', label='Limit')
        plt.plot([0.55,0.55],[-0.1, 1.05], 'r')
        plt.plot([-0.55,0.55],[1.05, 1.05], 'r')
        plt.plot([-0.55,0.55],[-0.1, -0.1], 'r')
    plt.legend()
    if impossible:
        plt.title("Quadrotor Tracking")
    else:
        plt.title("Quadrotor Tracking")
    plt.xlabel('X position (m)')
    plt.ylabel('Z position (m)')
    save_str = os.path.join(dir, 'quad_traj.png')
    plt.savefig(save_str)

def avg_rmse_plot(metric_data, save_dir, dt):
    # How to properly extract training time? Also, what counts as traiing time? Number of trajectories? Or number of
    # points actually used?
    avg_rmse = metric_data['average_rmse']
    steps = metric_data['average_length']
    std = metric_data['rmse_std']
    steps[0] = 0.0 # Linear MPC has no training

    t = 0
    times = [t]
    for i in range(1, len(steps)):
        t += steps[i]*dt
        times.append(t)

    times = np.array([times]).T
    fig, ax = plt.subplots()
    ax.plot(times, avg_rmse, '.')
    ax.errorbar(times, avg_rmse, yerr=std)
    ax.set_ylabel('XZ RMSE (m)')
    ax.set_xlabel('Training Times (s)')
    ax.set_title('Data Efficiency')

    fig.savefig(os.path.join(save_dir, 'test_rmses.png'))

def make_csvs(metric_data, dt, save_dir):
    avg_rmse = np.array([metric_data['average_rmse']]).T
    rmse = np.vstack(metric_data['rmse'])
    std = np.array([metric_data['rmse_std']]).T
    steps = np.array(metric_data['average_length'])

    steps[0] = 0.0 # Linear MPC has no training

    t = 0
    times = [t]
    for i in range(1, len(steps)):
        t += steps[i]*dt
        times.append(t)

    times = np.array([times]).T

    data = np.hstack((times, rmse, avg_rmse, std))
    headers = 'time'
    for i in range(rmse.shape[1]):
        headers += f',Test {i} RMSE'
    headers += ',Average RMSE'
    headers += ',RMSE std'
    np.savetxt(os.path.join(save_dir,'test_data.csv'), data, delimiter=",", header=headers)



class GPMPCExp(EpochExp):
    def launch_single_train_epoch(self,
                                  run_ctrl,
                                  env,
                                  n_episodes,
                                  episode_i,
                                  n_steps,
                                  log_freq,
                                  **kwargs):
        """Defined per controller?"""
        # Training Data Collection
        traj_data = self._execute_task(ctrl=run_ctrl,
                                       env=env,
                                       n_episodes=n_episodes,
                                       n_steps=n_steps,
                                       log_freq=log_freq)
        self.add_to_all_train_data(traj_data)
        # Parsing of training Data.
        train_inputs, train_outputs = self.preprocess_training_data(traj_data, episode_i, **kwargs)
        # Learning of training data.
        self.train_controller(train_inputs, train_outputs, **kwargs)

        return traj_data

    def train_controller(self, train_inputs, train_outputs, **kwargs):
        _ = self.ctrl.learn(input_data=train_inputs, target_data=train_outputs)

    def preprocess_training_data(self,
                                 traj_data,
                                 episode_i,
                                 num_samples,
                                 rand_kernel_selection):
        """
        Args:
            traj_data:
            rand_kernel_selection:
            episode_i:
            num_samples:

        """
        if rand_kernel_selection:
            x_seq, actions, x_next_seq = gather_training_samples(self.all_train_data,
                                                                 episode_i,
                                                                 num_samples,
                                                                 self.train_env.np_random)
        else:
            x_seq, actions, x_next_seq = gather_training_samples(traj_data, episode_i, num_samples)
        train_inputs, train_outputs = self.ctrl.preprocess_training_data(x_seq, actions, x_next_seq)
        return train_inputs, train_outputs


def main(config):
    env_func = partial(make,
                       config.task,
                       seed=config.seed,
                       **config.task_config
                       )
    config.algo_config.output_dir = config.output_dir
    ctrl = make(config.algo,
                env_func,
                seed=config.seed,
                **config.algo_config
                )
    ctrl.reset()

    num_epochs = config.num_epochs
    num_train_episodes_per_epoch = config.num_train_episodes_per_epoch
    num_test_episodes_per_epoch = config.num_test_episodes_per_epoch
    num_samples = config.num_samples

    train_envs = []
    for epoch in range(num_epochs):
        train_envs.append(env_func(randomized_init=False))
        train_envs[epoch].action_space.seed(config.seed)
    test_envs = []
    for epoch in range(num_epochs+1):
        test_envs.append(env_func(randomized_init=False))
        test_envs[epoch].action_space.seed(config.seed)
    exp =GPMPCExp(test_envs,
                  ctrl,
                  train_envs,
                  num_epochs,
                  num_train_episodes_per_epoch,
                  num_test_episodes_per_epoch,
                  config.output_dir,
                  save_train_trajs=True,
                  save_test_trajs=True)
    ref = exp.env.X_GOAL
    metrics, train_data, test_data, = exp.launch_training(num_samples=num_samples,
                                                          rand_kernel_selection=config.rand_kernel_selection)
    return train_data, test_data, metrics, ref, exp

if __name__ == "__main__":
    fac = ConfigFactory()
    fac.add_argument("--plot_dir", type=str, default='', help="Create plot from CSV file.")
    config = fac.merge()
    set_dir_from_config(config)
    mkdirs(config.output_dir)

    # Save config.
    with open(os.path.join(config.output_dir, 'config.yaml'), "w") as file:
        yaml.dump(munch.unmunchify(config), file, default_flow_style=False)

    train_data, test_data, metrics, ref, exp = main(config)

    #data = np.load(
    #    '/home/ahall/Documents/UofT/code/ahall_scg/experiments/arxiv/quadrotor_constraint/utils/temp-data/quad_impossible_traj_10hz/seed1337_Aug-24-16-15-24_v0.5.0-303-gcaa86b3/traj_data.npz',
    #    allow_pickle=True)
    #traj_data = data['traj_data'].item()
    #ref = traj_data.state[0]
    make_traking_plot(test_data, ref, config.output_dir)
    make_csvs(metrics, 1/config.task_config.ctrl_freq, config.output_dir)
    avg_rmse_plot(metrics, config.output_dir, 1/config.task_config.ctrl_freq)
