import utils
from utils.plots import *
import numpy as np
import matplotlib.pyplot as plt

#%%
# Kymogram
K = utils.Kymogram(sim_time=5)
beg_turn = []
end_turn = []
for i in range(K.n_timestamps):
    if i == 40:
        K.turn[i, 0] = 1
    if i == 70:
        K.turn[i, 0] = 0
    if i == 80:
        K.omega_modification(-K.omega, i)
    K.kymogram[i] = K.step(i)
    beg_turn.append(K.begin_turn_position)
    end_turn.append(K.end_turn_position)

#%%
fig, ax = plt.subplots()
ax.plot(np.arange(K.n_timestamps), beg_turn)
ax.plot(np.arange(K.n_timestamps), end_turn)

#%%
# Kymogram figure
matrix = K.kymogram.transpose()
shape = np.shape(matrix)

fig, ax = plt.subplots(dpi=120)
im = ax.imshow(matrix, aspect=.3 * (shape[1] / shape[0]), cmap='bwr')
ax.set_title('Kymogram')
ax.set_xlabel('frame number')
ax.set_ylabel('body position')
ax.set_yticks([0, shape[0] - 1])
ax.set_yticklabels(['head', 'tail'])

[l, b, w, h] = ax.get_position().bounds
cbound = [l + w * 1.05, b, w * 0.05, h]
cax = fig.add_axes(cbound)
cbar = fig.colorbar(im, ax=ax, cax=cax)
cax.set_title('dorsal', fontsize=8)
cax.set_xlabel('ventral', fontsize=8)

set_bbox_inches_tight(fig)
plt.show()

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
