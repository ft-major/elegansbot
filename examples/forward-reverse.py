from elegansbot import Worm, set_bbox_inches_tight
import numpy as np
import matplotlib.pyplot as plt


class Kymogram(object):
    def __init__(self, sim_time=5):
        self.fps = 30
        self.timestamps = self.fps * sim_time
        self.n_rods = 25
        self.kymogram = np.zeros([self.timestamps, self.n_rods - 1])
        self.nu = 1.832 * np.ones(self.n_rods - 1)
        self.omega = - 2 * np.pi / 1.16 * np.ones(self.n_rods - 1)
        self.A = 0.5 * np.ones(self.n_rods - 1)
        self.phi = self.phi_forward()

    def phi_forward(self):
        return 2 * np.pi * self.nu * (np.arange(self.n_rods - 1) / (self.n_rods - 2))

    def forward(self, t):
        return self.A * np.cos(self.omega * t + self.phi)


#%%
# Kymogram
K = Kymogram(sim_time=5)
kymo = np.zeros((K.timestamps, K.n_rods - 1))
for i in range(K.timestamps):
    t = (1 / K.fps) * i
    K.kymogram[i] = K.forward(t)
    if t == 2:
        K.phi = K.phi - 2 * K.omega * t
        K.omega = - K.omega

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
env = Worm(scale_friction=0.01)
env.run(K.kymogram, 1 / K.fps)

# env.plot_overview()
# env.plot_speed_graph()
# %matplotlib notebook # uncomment this line if the code runs in jupyter-notebook.
env.play_animation(speed_playback=0.5)
# %matplotlib inline # uncomment this if it's in jupyter-notebook.
# env.save_animation('demo.mp4') # FFMPEG is required for saving animation.
