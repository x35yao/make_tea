from TP_PMP import utils
from TP_PMP import promp_gmm as promp
from TP_PMP import promp_gaussian as promp_gaussian
import numpy as np
import random


class PMP():
    def __init__(self, data_in_all_rfs, times, dof, dim_basis_fun = 25, sigma = 0.035, full_basis = None, n_components = 1, covariance_type = 'block_diag', max_iter = 100, n_init = 1, gmm = True):
        self.sigma = sigma
        self.dof = dof
        if full_basis == None:
            self.full_basis = {
            'conf': [
                {"type": "sqexp", "nparams": 22, "conf": {"dim": 21}},
                {"type": "poly", "nparams": 0, "conf": {"order": 3}}
            ],
            'params': [np.log(self.sigma), 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65
                , 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
        }
        self.dim_basis_fun = dim_basis_fun
        self.data_in_all_rfs = data_in_all_rfs
        self.rfs = sorted(data_in_all_rfs.keys())
        self.times = times
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.inv_whis_mean = lambda Sigma: utils.make_block_diag(Sigma, self.dof * len(self.rfs))
        self.prior_Sigma_w = {'v': self.dim_basis_fun * self.dof, 'mean_cov_mle': self.inv_whis_mean}
        self.prior_mu_w = {'k0':2, 'm0':0}
        self.prior_mu_w = None
        self.max_iter = max_iter
        self.n_init = n_init
        self.n_demos = len(self.times)
        self.gmm = gmm
        if self.gmm:
            self.pmp = promp.FullProMP(basis=self.full_basis, n_dims= len(self.rfs) * self.dof, n_rfs = len(self.rfs), n_components=self.n_components,
                                        covariance_type=self.covariance_type)
        else:
            self.pmp = promp_gaussian.FullProMP(basis=self.full_basis, n_dims=len(self.rfs) * self.dof, n_rfs=len(self.rfs))

    def train(self, print_lowerbound = True, no_Sw = False):
        data_concat = self._concat_data_across_rfs()
        train_summary = self.pmp.train(self.times, data=data_concat, print_lowerbound=print_lowerbound, no_Sw=no_Sw,
                                        max_iter=self.max_iter, prior_Sigma_w=self.prior_Sigma_w, prior_mu_w = self.prior_mu_w,
                                        n_init=self.n_init)
    def refine(self, max_iter = None):
        if max_iter is None:
            max_iter = self.max_iter
        self.pmp.refine(max_iter)

    def bic(self):
        return self.pmp.bic()
    def _concat_data_across_rfs(self):
        '''
        For each demonstration, this function concatenate data for all reference frames
        '''
        data_concat = []
        rfs = sorted(self.data_in_all_rfs.keys())
        for i in range(self.n_demos):
            temp = np.hstack([self.data_in_all_rfs[rf][i] for rf in rfs])
            data_concat.append(temp)
        return data_concat

    def select_mode(self):
        modes = np.arange(self.n_components)
        selected_mode_ind = random.choices(modes, weights = self.promp.alpha, k = 1)[0]
        return selected_mode_ind

    def marginal_w(self, t):
        selected_mode_ind = self.select_mode()
        model = self.pmp
        mu,sigma = model.marginal_w(t, selected_mode_ind)

        return mu, sigma

    def condition(self, t, T, q, Sigma_q=None, ignore_Sy = True):
        self.pmp.condition(t, T, q, Sigma_q, ignore_Sy)
