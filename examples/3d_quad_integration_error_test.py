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
import pybullet as p

from safe_control_gym.utils.utils import str2bool
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import PIDController
from safe_control_gym.math_and_models.transformations import draw_frame, RotXYZ
from safe_control_gym.controllers.mpc.mpc_utils import rk_discrete
from utils import rmse

"""The main function creating, running, and closing an environment.

"""
CONFIG_FACTORY = ConfigFactory()
config = CONFIG_FACTORY.merge()
freqs = config.freqs
all_rmse = []
for freq in freqs:
    # Set iterations and episode counter.
    num_episodes = 1
    #ITERATIONS = int(1000)
    # Start a timer.
    START = time.time()

    # Create an environment
    #config.quadrotor_config.ctrl_freq = freq
    config.quadrotor_config.pyb_freq = freq
    env = make('quadrotor', **config.quadrotor_config)

    ITERATIONS = int(config.quadrotor_config.ctrl_freq)

    all_obs = np.zeros((env.state_dim, ITERATIONS+1))
    all_obs_w_body_rates = np.zeros((env.state_dim, ITERATIONS+1))
    all_obs_sym = np.zeros((env.state_dim, ITERATIONS+1))

    # Discrete symbolic dynamics
    fd_sim = rk_discrete(env.symbolic.fc_func, env.state_dim, env.action_dim, 1.0/freq)

    # Controller
    ctrl = PIDController()

    # Reset the environment, obtain and print the initial observations.
    initial_obs, initial_info = env.reset()
    p.setPhysicsEngineParameter(numSubSteps=int(config.pbsubsteps),
                                #numSolverIterations=1000,
                                #solverResidualThreshold=1e-9,
                                physicsClientId=env.PYB_CLIENT)
    #p.changeDynamics(0, -1, linearDamping=0, angularDamping=0)
    all_obs[:,0] = initial_obs
    all_obs_sym[:,0] = initial_obs
    # Dynamics info
    print('\nPyBullet dynamics info:')
    print('\t' + str(p.getDynamicsInfo(bodyUniqueId=env.DRONE_IDS[0], linkIndex=-1, physicsClientId=env.PYB_CLIENT)))
    print('\nInitial reset.')
    print('\tInitial observation: ' + str(initial_obs))

    # Run an experiment.
    obs = initial_obs
    obs_sym = initial_obs
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
                        target_pos=np.array([0.5, 0.5, 0.5]),
                        target_rpy=np.array([0.0, 0.0, np.pi/2.0])
            )
            action = rpms
            action = env.KF * action**2
        #action = np.zeros((4,))
        #obs_sym[6:9] = obs[6:9]
        #obs_sym[9:] = obs[9:]
        #obs_sym[9:] = RotXYZ(obs[6], obs[7], obs[8]) @ obs[9:]

        obs, reward, done, info = env.step(action)
        all_obs[:, i+1] = obs
        # Step the environment and print all returned information.
        for _ in range(int(freq/config.quadrotor_config.ctrl_freq)):
            obs_sym = fd_sim(x0=obs_sym, p=env.forces)['xf'].toarray()[:,0]
        #obs_sym = env.symbolic.fd_func(x0=obs, p=action)['xf'].toarray()[:, 0]
        all_obs_sym[:, i+1] = obs_sym
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


    # Close the environment and print timing statistics.
    env.close()
    elapsed_sec = time.time() - START
    out = str("\n{:d} iterations (@{:d}Hz) and {:d} episodes in {:.2f} seconds, i.e. {:.2f} steps/sec for a {:.2f}x speedup.\n\n"
          .format(ITERATIONS, env.CTRL_FREQ, num_episodes, elapsed_sec, ITERATIONS/elapsed_sec, (ITERATIONS*env.CTRL_TIMESTEP)/elapsed_sec))
    print(out)

    # All obs in body rates
    for i in range(ITERATIONS+1):
        all_obs_w_body_rates[:,i] = all_obs[:,i]
        all_obs_w_body_rates[9:,i] = RotXYZ(all_obs[6, i], all_obs[7, i], all_obs[8, i]).T @ all_obs[9:, i]

    rmses = rmse(all_obs, all_obs_sym)
    all_rmse.append(rmses[:,None])
    for i in range(env.state_dim):
        print(f'{env.STATE_LABELS[i]} RMSE: {rmses[i]}')

    if config.plot_traj:
        diff = all_obs - all_obs_sym
        abs_diff = np.abs(diff)
        for i in range(env.state_dim):
            print(f'{env.STATE_LABELS[i]} Mean Absolute difference: {np.mean(abs_diff[i, :])}')
            print(f'{env.STATE_LABELS[i]} STD Absolute Diff: {np.std(abs_diff[i, :])}')
            fig, ax = plt.subplots()
            ax.plot(all_obs_w_body_rates[i, :], label='pybullet')
            ax.plot(all_obs_sym[i, :], label='sym')
            ax.set_xlabel('Step')
            ax.set_title(f'{env.STATE_LABELS[i]}')
            ax.legend()
            # ax.hist(diff[i,:],bins=50)
            # ax.set_xlabel('Error')
            # ax.set_ylabel('counts')
            # ax.set_title(f'{env.STATE_LABELS[i]} Error')
            plt.show()
all_rmse = np.hstack(all_rmse)

if config.plot_rmse:
    n_rmse = len(all_rmse)
    for dim in range(env.state_dim):
        fig, ax = plt.subplots()
        ax.set_title(f'{env.STATE_LABELS[dim]} RMSE')
        ax.plot(freqs, all_rmse[dim,:])
        ax.set_xlabel('Freq')
        ax.set_ylabel('RMSE')
print(f'Final Pybulet position: {all_obs[4,-1]}')
print(f'Final Symoblic position: {all_obs_sym[4,-1]}')
