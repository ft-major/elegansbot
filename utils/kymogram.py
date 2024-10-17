import numpy as np


class Kymogram(object):
    def __init__(self, sim_time=5):
        self.fps = 30
        self.n_timestamps = self.fps * sim_time
        self.timestamps = np.arange(0, self.n_timestamps)
        self.t = np.linspace(0, sim_time, self.n_timestamps)
        self.n_rods = 25
        self.kymogram = np.zeros([self.n_timestamps, self.n_rods - 1])
        self.nu = 1.832  # * np.ones(self.n_rods - 1)
        self.omega = - 2 * np.pi / 1.16  # * np.ones(self.n_rods - 1)
        self.A = 0.5  # * np.ones(self.n_rods - 1)
        self.phi = self.phi_forward()
        self.turn = np.zeros([self.n_timestamps, self.n_rods - 1])
        self.turn_propagation = (self.omega * self.dt) / (self.phi[0] - self.phi[1])
        self.begin_turn_position = self.n_rods - 1
        self.end_turn_position = self.n_rods - 1

    @property
    def dt(self):
        return 1 / self.fps

    def phi_forward(self):
        return 2 * np.pi * self.nu * (np.arange(self.n_rods - 1) / (self.n_rods - 2))

    def phi_turn(self, rod):
        self.phi[rod] -= self.omega * self.dt

    def step(self, timestamp):
        if np.any(self.turn[timestamp, :]):
            for joint in np.arange(self.n_rods - 1):
                if self.turn[timestamp, joint] == 1:
                    self.phi_turn(joint)
                if joint == 0:
                    if self.turn[timestamp, joint] != self.turn[timestamp - 1, joint]:
                        if self.turn[timestamp, joint] == 1:
                            self.begin_turn_position = 0
                        elif self.turn[timestamp, joint] == 0:
                            self.end_turn_position = 0
                elif joint == self.n_rods - 2:
                    if self.turn[timestamp, joint] != self.turn[timestamp - 1, joint]:
                        if self.turn[timestamp, joint] == 1:
                            self.begin_turn_position = self.n_rods - 1
                        elif self.turn[timestamp, joint] == 0:
                            self.end_turn_position = self.n_rods - 1
            if self.begin_turn_position < self.n_rods - 1:
                self.begin_turn_position += self.turn_propagation
                for i in range(0, int(np.rint(self.begin_turn_position)) + 1):
                    self.turn[timestamp + 1, i] = 1
            else:
                if self.end_turn_position < self.n_rods - 1:
                    self.turn[timestamp + 1, :] = 1
            if self.end_turn_position < self.n_rods - 1:
                self.end_turn_position += self.turn_propagation
                for i in range(0, int(np.rint(self.end_turn_position)) + 1):
                    self.turn[timestamp + 1, i] = 0
        return self.A * np.cos(self.omega * self.t[timestamp] + self.phi)

    def omega_modification(self, new_omega, timestamp):
        self.phi += (self.omega - new_omega) * self.t[timestamp]
        self.turn_propagation *= self.omega / new_omega
        self.omega = new_omega
