#!/bin/bash

# Run GPMPC with learned model
python3 impossible_traj.py --algo safe_explorer_ppo --task quadrotor --overrides ./data/config.yaml
