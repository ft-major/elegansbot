import utils
from utils.plots import *
import numpy as np

fps = 32
kymogram = np.load('kymogram_omega-turn.npy')

env = utils.Worm(scale_friction=0.01)
env.run(kymogram, 1/fps)
play_animation(env)