from TP_PMP import utils
from TP_PMP import full_promp as promp
import numpy as np

class PMP():
    def __init__(self, Q, times, dof, dim_basis_fun = 30, sigma = 0.035, full_basis = None):
        self.sigma = sigma
        self.dof = dof
        if full_basis == None:
            self.full_basis = {
            'conf': [
                {"type": "sqexp", "nparams": 22, "conf": {"dim": 21}},
                {"type": "poly", "nparams": 0, "conf": {"order": 1}},
                {"type": "poly", "nparams": 0, "conf": {"order": 2}},
                {"type": "poly", "nparams": 0, "conf": {"order": 3}}
            ],
            'params': [np.log(self.sigma), 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65
                , 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
        }
        self.dim_basis_fun = dim_basis_fun
        self.Q = Q
        self.times = times
        self.promp = promp.FullProMP(basis= self.full_basis)
        self.inv_whis_mean = lambda v, Sigma: 9e-1 * utils.make_block_diag(Sigma, self.dof) + 1e-1 * np.eye(self.dof * self.dim_basis_fun)
        self. prior_Sigma_w = {'v': self.dim_basis_fun * self.dof, 'mean_cov_mle': self.inv_whis_mean}
    def train(self):
        train_summary = self.promp.train(self.times, q=self.Q, max_iter=30, prior_Sigma_w= self.prior_Sigma_w,
                                                 print_inner_lb=False)

    def marginal_w(self, t):
        mu, sigma = self.promp.marginal_w(t)
        return mu, sigma
