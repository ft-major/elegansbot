# -*- coding: utf-8 -*-
# %%
import utils
from utils.plots import *
import numpy as np

# %%
K = utils.Kymogram(sim_time=20)
env = utils.Worm(dt=0.0001, dt_snapshot=1 / K.fps, simTime=K.simTime)  # Numba JIT spends compile-time here, once.

#%%
motor_command = []
for i in range(env.n_snapshot):
    percentage = i / env.n_snapshot * 100
    if percentage < 15:
        motor_command.append("forward")
    elif percentage < 17:
        motor_command.append("turn")
    elif percentage < 39:
        motor_command.append("forward")
    elif percentage < 42:
        motor_command.append("turn")
    elif percentage < 55:
        motor_command.append("forward")
    elif percentage < 65:
        motor_command.append("reverse")
    elif percentage < 75:
        motor_command.append("forward")
    else:
        motor_command.append("slowing")

    K.motor_command(i, motor_command[i])
    action = K.kymogram[i]
    env.act = action
    env.steps()

#%%
# plot_kymogram(K.kymogram)
# plot_overview(env)
# plot_speed_graph(env)
play_animation(env, motor_command=motor_command)
# save_animation(env, 'demo_crawl.mp4')
