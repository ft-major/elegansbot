# -*- coding: utf-8 -*-
# %%
import utils
from utils.plots import *
import numpy as np

# %%
K = utils.Kymogram(sim_time=10)
env = utils.Worm(dt=0.0001, dt_snapshot=1 / K.fps, simTime=K.simTime)  # Numba JIT spends compile-time here, once.
for i in range(env.n_snapshot):
    if 40 < i < 48:
        K.motor_command(i, "slowing")
    elif i == 49:
        K.motor_command(i, "forward")
    elif 50 < i < 80:
        K.motor_command(i, "turn")
    elif 90 < i < 100:
        K.motor_command(i, "reverse")
    elif i == 110:
        K.motor_command(i, "forward")
    # elif 130 < i < 140:
    #     K.motor_command(i, "slowing")
    elif i == 160:
        K.motor_command(i, "reverse")
    elif i == 200:
        K.motor_command(i, "forward")
    else:
        K.motor_command(i)
    action = K.kymogram[i]
    env.act = action
    env.steps()
plot_kymogram(K.kymogram)
# plot_overview(env)
# plot_speed_graph(env)
play_animation(env)
# save_animation(env, 'demo_crawl.mp4')
