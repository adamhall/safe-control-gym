"""3D quadrotor example script.

Example:

    $ python3 3d_quad.py --overrides ./3d_quad.yaml

"""
import time
import yaml
import inspect
import numpy as np
import pybullet as p
import casadi as cs
import matplotlib.pyplot as plt

from safe_control_gym.utils.utils import str2bool
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import PIDController
from safe_control_gym.math_and_models.transformations import draw_frame, RotXYZ

"""The main function creating, running, and closing an environment.

"""
# Set iterations and episode counter.
num_episodes = 1
#ITERATIONS = int(1000)
# Start a timer.
START = time.time()

# Create an environment
CONFIG_FACTORY = ConfigFactory()
config = CONFIG_FACTORY.merge()
env = make('quadrotor', **config.quadrotor_config)
ITERATIONS = int(config.quadrotor_config.ctrl_freq)

all_obs = np.zeros((env.state_dim, ITERATIONS+1))
all_obs_sym = np.zeros((env.state_dim, ITERATIONS+1))

# Controller
ctrl = PIDController()

# Reset the environment, obtain and print the initial observations.
initial_obs, initial_info = env.reset()
all_obs[:,0] = initial_obs
all_obs_sym[:,0] = initial_obs
# Dynamics info
print('\nPyBullet dynamics info:')
print('\t' + str(p.getDynamicsInfo(bodyUniqueId=env.DRONE_IDS[0], linkIndex=-1, physicsClientId=env.PYB_CLIENT)))
print('\nInitial reset.')
print('\tInitial observation: ' + str(initial_obs))

# Run an experiment.
obs = initial_obs
for i in range(ITERATIONS):
    # Step by keyboard input
    # _ = input('Press any key to continue.')
    # Sample a random action.
    if i < 0:
        action = env.action_space.sample()
    else:
        rpms, _, _ = ctrl.compute_control(control_timestep=env.CTRL_TIMESTEP,
                    cur_pos=np.array([obs[0],obs[2],obs[4]]),
                    cur_quat=np.array(p.getQuaternionFromEuler([obs[6],obs[7],obs[8]])),
                    cur_vel=np.array([obs[1],obs[3],obs[5]]),
                    cur_ang_vel=np.array([obs[9],obs[10],obs[11]]),
                    #target_pos=np.array([0.1, 0.1, 0.5]),
                    target_pos=np.array([0.1, 0.1, 0.1]),
                    target_rpy=np.array([0.0, 0.0, np.pi/2.0])
        )
        action = rpms
        action = env.KF * action**2
    # Step the environment and print all returned information.
    if i == 0:
        obs_sym = env.symbolic.fd_func(x0=initial_obs, p=action)['xf'].toarray()[:,0]
    else:
        obs_sym = env.symbolic.fd_func(x0=obs_sym, p=action)['xf'].toarray()[:,0]
        #obs_sym = env.symbolic.fd_func(x0=obs, p=action)['xf'].toarray()[:, 0]
    all_obs_sym[:, i+1] = obs_sym
    obs, reward, done, info = env.step(action)
    all_obs[:, i+1] = obs
    #
    print('\n'+str(i)+'-th step.')
    out = '\tApplied action: ' + str(action)
    print(out)
    out = '\tObservation: ' + str(obs)
    print(out)
    out = '\tReward: ' + str(reward)
    print(out)
    out = '\tDone: ' + str(done)
    print(out)
    # out = '\tConstraints evaluations: ' + str(info['constraint_values'])
    # print(out)
    # out = '\tConstraints violation: ' + str(bool(info['constraint_violation']))
    # print(out)

    # If an episode is complete, reset the environment.
    if done:
        num_episodes += 1
        new_initial_obs, new_initial_info = env.reset()
        # print(str(num_episodes)+'-th reset.', 7)
        # print('Reset obs' + str(new_initial_obs), 2)
        # print('Reset info' + str(new_initial_info), 0)

# Close the environment and print timing statistics.
env.close()
elapsed_sec = time.time() - START
out = str("\n{:d} iterations (@{:d}Hz) and {:d} episodes in {:.2f} seconds, i.e. {:.2f} steps/sec for a {:.2f}x speedup.\n\n"
      .format(ITERATIONS, env.CTRL_FREQ, num_episodes, elapsed_sec, ITERATIONS/elapsed_sec, (ITERATIONS*env.CTRL_TIMESTEP)/elapsed_sec))
print(out)

diff = all_obs - all_obs_sym
abs_diff = np.abs(diff)
for i in range(env.state_dim):
    print(f'{env.STATE_LABELS[i]} Mean Absolute difference: {np.mean(abs_diff[i,:])}')
    print(f'{env.STATE_LABELS[i]} STD Absolute Diff: {np.std(abs_diff[i,:])}')
    fig, ax = plt.subplots()
    ax.plot(all_obs[i,:], label='pybullet')
    ax.plot(all_obs_sym[i,:], label='sym')
    ax.set_xlabel('Step')
    ax.set_title(f'{env.STATE_LABELS[i]}')
    ax.legend()
    #ax.hist(diff[i,:],bins=50)
    #ax.set_xlabel('Error')
    #ax.set_ylabel('counts')
    #ax.set_title(f'{env.STATE_LABELS[i]} Error')
    plt.show()
p.getMatrixFromQuaternion(q)
