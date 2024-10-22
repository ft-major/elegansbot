import utils
from utils.plots import *
import numpy as np
import matplotlib.pyplot as plt

#%%
# Kymogram
K = utils.Kymogram(sim_time=5)
for i in range(K.n_timestamps):
    if i == 40:
        K.turn[i, 0] = 1
    if i == 60:
        K.turn[i, 0] = 0
    if i == 80:
        K.omega_modification(-K.omega, i)
    K.kymogram[i] = K.step(i)

#%%
# Kymogram figure
plot_kymogram(K.kymogram)

# %%

# ElegansBot Simulation
env = utils.Worm(scale_friction=0.01)
env.run(K.kymogram, 1 / K.fps)

# env.plot_overview()
# plot_speed_graph(env)
# %matplotlib notebook # uncomment this line if the code runs in jupyter-notebook.
play_animation(env, speed_playback=0.5)
# %matplotlib inline # uncomment this if it's in jupyter-notebook.
# env.save_animation('demo.mp4') # FFMPEG is required for saving animation.
