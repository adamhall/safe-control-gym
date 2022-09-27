'''A quadrotor trajectory tracking example.

Notes:
    Includes and uses PID control.

Run as:
    $ python3 tracking.py --algo pid --task quadrotor --overrides ./tracking.yaml
'''

from functools import partial

from safe_control_gym.experiments.experiment import Experiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

def run(gui=True, n_episodes=1, n_steps=None):
    '''The main function running the PID trajectory tracking quadrotor example.

    Args:
        gui (bool): Whether to display the gui and plot graphs.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): The total number of steps to execute.
    '''

    # Create an environment
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge()

    for i in range(3):
        if i == 1:
            config.task_config['task_info']['trajectory_type'] = 'circle'
        elif i == 2:
            config.task_config['task_info']['trajectory_type'] = 'square'

        # Create controller.
        env_func = partial(make,
                        config.task,
                        **config.task_config
                        )
        env = env_func(gui=gui)
        ctrl = make(config.algo,
                    env_func,
                    )

        # Run the experiment.
        experiment = Experiment(env, ctrl)
        trajs_data, _ = experiment.run_evaluation(n_episodes=n_episodes, n_steps=n_steps)
        experiment.close()

        iterations = len(trajs_data['action'][0])
        for i in range(iterations):
            # Step the environment and print all returned information.
            obs, reward, done, info, action = trajs_data['obs'][0][i], trajs_data['reward'][0][i], trajs_data['done'][0][i], trajs_data['info'][0][i], trajs_data['action'][0][i]

            # Print the last action and the information returned at each step.
            print(i, '-th step.')
            print(action, '\n', obs, '\n', reward, '\n', done, '\n', info, '\n')

        elapsed_sec = trajs_data['timestamp'][0][-1] - trajs_data['timestamp'][0][0]
        print('\n{:d} iterations (@{:d}Hz) and {:d} episodes in {:.2f} seconds, i.e. {:.2f} steps/sec for a {:.2f}x speedup.\n'
            .format(iterations, config.task_config.ctrl_freq, 1, elapsed_sec, iterations/elapsed_sec, (iterations*(1. / config.task_config.ctrl_freq))/elapsed_sec))

if __name__ == '__main__':
    run()
