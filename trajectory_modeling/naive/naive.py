import numpy as np
from scipy.interpolate import interpn
class Naive():
    def __init__(self):
        pass
    def train(self, data, times):
        self.time = np.mean(times, axis = 0)
        self.mu = []
        self.sigma = []
        data = np.array(data)
        for t in range(data.shape[1]):
            data_t = data[:, t, :]
            self.mu.append(np.mean(data_t, axis=0))
            self.sigma.append(np.cov(data_t.T))
        self.mu = np.array(self.mu)
        self.sigma = np.array(self.sigma)


    def predict(self, t):
        D = self.mu.shape[1]
        N = len(t)
        mu = np.zeros((N, D))
        sigma = np.zeros((N,D,D))
        for i in range(D):
            mu_i = np.interp(t, self.time, self.mu[:, i])
            mu[:,i] = mu_i
            for j in range(D):
                sigma_ij = np.interp(t, self.time, self.sigma[:, i, j])
                sigma[:,i, j] = sigma_ij
        return mu, sigma
