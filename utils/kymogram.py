import numpy as np


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

    def forward(self, _t):
        return self.A * np.cos(self.omega * _t + self.phi)

    def omega_modification(self, new_omega, _t):
        self.phi = self.phi + self.omega * _t - new_omega * _t
        self.omega = new_omega
