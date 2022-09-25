"""Example Google style docstrings.

This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

"""
import os
import munch
import yaml
import shutil
import torch
import matplotlib.pyplot as plt
from munch import munchify
import numpy as np
import pybullet as p
from functools import partial
from PIL import Image, ImageGrab

from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import read_file
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.utils import mkdirs, set_dir_from_config


def main(config):
    env_func = partial(make,
                       config.task,
                       **config.task_config
                       )

    # Create controller from PPO YAML.
    ppo_config_dir = os.path.dirname(os.path.abspath(__file__))+'/data'
    ppo_dict = read_file(os.path.join(ppo_config_dir,'config.yaml'))
    ppo_config = munchify(ppo_dict)
    # Setup PPO controller.
    ctrl = make(ppo_config.algo,
                    env_func,
                    **ppo_config.algo_config)
    # Load state_dict from trained PPO.
    ppo_model_dir = os.path.dirname(os.path.abspath(__file__))+'/data'
    ctrl.load(os.path.join(ppo_model_dir,'model_latest.pt'))  # Show violation.
    # Remove temporary files and directories
    shutil.rmtree(os.path.dirname(os.path.abspath(__file__))+'/temp')

    ctrl.reset()

    env = env_func(gui=True,
                   randomized_init=False,
                   info_on_reset=True)

    initial_obs, initial_info = env.reset()
    # Plot constraint
    p.addUserDebugLine(lineFromXYZ=[-0.45, 0, 0],
                       lineToXYZ=[0.45, 0, 0],
                       lineColorRGB=[0, 0, 0],
                       # lifeTime=2 * env._CTRL_TIMESTEP,
                       physicsClientId=env.PYB_CLIENT)
    p.addUserDebugLine(lineFromXYZ=[-0.45, 0, 0.9],
                       lineToXYZ=[0.45, 0, 0.9],
                       lineColorRGB=[0, 0, 0],
                       # lifeTime=2 * env._CTRL_TIMESTEP,
                       physicsClientId=env.PYB_CLIENT)
    p.addUserDebugLine(lineFromXYZ=[0.45, 0, 0],
                       lineToXYZ=[0.45, 0, 0.9],
                       lineColorRGB=[0, 0, 0],
                       # lifeTime=2 * env._CTRL_TIMESTEP,
                       physicsClientId=env.PYB_CLIENT)
    p.addUserDebugLine(lineFromXYZ=[-0.45, 0, 0],
                       lineToXYZ=[-0.45, 0, 0.9],
                       lineColorRGB=[0, 0, 0],
                       # lifeTime=2 * env._CTRL_TIMESTEP,
                       physicsClientId=env.PYB_CLIENT)
    # Plot trajectory.
    for i in range(0, initial_info['x_reference'].shape[0]-1):
        p.addUserDebugLine(lineFromXYZ=[initial_info['x_reference'][i, 0], 0,
                                        initial_info['x_reference'][i, 2]],
                           lineToXYZ=[initial_info['x_reference'][i+1, 0], 0, initial_info['x_reference'][i+1, 2]],
                           lineColorRGB=[1, 0, 0],
                           # lifeTime=2 * env._CTRL_TIMESTEP,
                           physicsClientId=env.PYB_CLIENT)

    ITERATIONS = int(config.task_config['episode_len_sec']*config.task_config['ctrl_freq'])
    # Run the experiment.
    obs = initial_obs
    c = initial_info['constraint_values']
    frames = []
    mon = (2770, 108, 3529, 909)
    for i in range(ITERATIONS):
    #for i in range(5):
        img = ImageGrab.grab(mon)
        frames.append(img)
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(ctrl.device)
            c = torch.FloatTensor(c).to(ctrl.device)
            action = ctrl.agent.ac.act(obs, c=c)
        # Step the environment and print all returned information.
        obs, reward, done, info = env.step(action)
        # Print the last action and the information returned at each step.
        print(i, '-th step.')
        print(action, '\n', obs, '\n', reward, '\n', done, '\n', info, '\n')
        # Compute the next action.
        obs = ctrl.obs_normalizer(obs)
        c = info["constraint_values"]
        #env.render()
        #frames.append(env.render("rgb_array"))
        if done:
            _, _ = env.reset()
    # Close the environment and print timing statistics.
    env.close()
    #imgs = [Image.fromarray(img) for img in frames]
    duration = 20
    # duration is the number of milliseconds between frames; this is 40 frames per second
    frames[0].save(os.path.join(config.output_dir,"array.gif"), save_all=True, append_images=frames[1:], duration=duration, loop=0)



if __name__ == "__main__":
    fac = ConfigFactory()
    fac.add_argument("--run_prior", type=bool, default=False, help="True to run only prior model (no learning)")
    config = fac.merge()
    set_dir_from_config(config)
    mkdirs(config.output_dir)

    # Save config.
    with open(os.path.join(config.output_dir, 'config.yaml'), "w") as file:
        yaml.dump(munch.unmunchify(config), file, default_flow_style=False)

    main(config)
