import utils
from utils.plots import *
import numpy as np

#%%
# Kymogram
K = utils.Kymogram(sim_time=20)
for i in K.timestamps:
    K.step(i)
    if 60 < i < 80 and i % 2:
        K.omega_modification(K.omega - 0.05 * K.initial_omega, i)
        print("slowing", i)
    if i == 120:
        K.omega_modification(-K.omega, i)
        print("reverse", i)
    if i == 200:
        K.omega_modification(-K.omega * 120 / 100, i)
        print("forward faster", i)

#%%
# Kymogram figure
plot_kymogram(K.kymogram)

# %%

# ElegansBot Simulation
env = utils.Worm(scale_friction=0.01)
env.run(K.kymogram, 1 / K.fps)

# env.plot_overview()
plot_speed_graph(env)
# %matplotlib notebook # uncomment this line if the code runs in jupyter-notebook.
play_animation(env, speed_playback=0.5)
# %matplotlib inline # uncomment this if it's in jupyter-notebook.
# env.save_animation('demo.mp4') # FFMPEG is required for saving animation.
