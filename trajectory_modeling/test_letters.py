import numpy as np
from scipy.io import loadmat # loading data from matlab
import random
import TP_GMM.gmm as gmm
import TP_GMM.plot as plot
from test import get_bic, marginal_t, get_aic, get_position_difference_per_step
from matplotlib import pyplot as plt
from TP_PMP import pmp

letter = 'K' # choose a letter in the alphabet
datapath = './TP_GMM/data/2Dletters/'
data = loadmat(datapath + '%s.mat' % letter)
demos = [d['pos'][0][0].T for d in data['demos'][0]] # cleaning awful matlab data
demos_with_t = []
for demo in demos:
    t = np.linspace(0,1, demo.shape[0])
    temp = np.c_[t, demo]
    demos_with_t.append(temp)
n_test = 20
dists_gmm, dists_pmp = [], []
for i in range(n_test):
    demo_inds = list(range(len(demos_with_t)))
    n_train = 3
    train_inds = random.sample(demo_inds, k=n_train)
    test_inds = [ind for ind in demo_inds if ind not in train_inds]
    train_demos = [demos_with_t[ind] for ind in train_inds]
    test_demos = [demos_with_t[ind] for ind in test_inds]
    test_demo = random.sample(test_demos, k = 1)[0]

    max_num_states = 50
    gmms = []
    bics = []
    aics = []
    for j in range(max_num_states):
        data_gmm = np.concatenate(train_demos)
        t = np.linspace(0,1, train_demos[0].shape[0])
        n_states = j + 1
        n_data = data_gmm.shape[0]
        n_dims = data_gmm.shape[1]
        model_gmm = gmm.GMM(nb_states=n_states, nb_dim=n_dims)
        model_gmm.em(data_gmm, reg=1e-3, maxiter=200, verbose=False)
        LL = model_gmm.LL
        bic = get_bic(LL,  n_states * n_dims + n_states * (n_dims * (n_dims + 1) / 2), n_data)
        aic = get_aic(LL, n_states * n_dims + n_states * (n_dims * (n_dims + 1) / 2), n_data)
        # mu, sigma = marginal_t(model_gmm, t)
        # plt.figure(figsize = (8, 8))
        # plt.plot(mu[:,0], mu[:,1])
        # plt.show()
        # mu = model_gmm.mu[:,1:]
        # sigma = model_gmm.sigma[:,1:, 1:]
        # plt.figure(figsize = (8, 8))
        # for p in train_demos:
        #     plt.plot(p[:,1], p[:,2])
        # plot.plot_gmm(mu, sigma)
        # plt.show()
        gmms.append(model_gmm)
        bics.append(bic)
        aics.append(aic)

    temp = []
    for g in gmms:
        t = np.linspace(0, 1, test_demo.shape[0])
        mu_gmm, sigma_gmm = marginal_t(g, t)
        ground_truth = test_demo[:,1:]
        average_dist = np.mean(get_position_difference_per_step(ground_truth, mu_gmm))
        temp.append(average_dist)
    # print(f'The distance for the best gmm model is : {np.min(temp)}, The best model is {np.argmin(temp)}')
    dist_gmm = np.min(temp)

    data_pmp = [demo[:, 1:] for demo in train_demos]
    times = [demo[:,0] for demo in train_demos]
    model_pmp = pmp.PMP(data_pmp, times, n_dims - 1 )
    model_pmp.train()
    mu_pmp, sigma_pmp = marginal_t(model_pmp, t)
    dist_pmp = np.mean(get_position_difference_per_step(ground_truth, mu_pmp))
    # print(f'The distance for different pmp model: {dist_pmp}')
    dists_gmm.append(dist_gmm)
    dists_pmp.append(dist_pmp)
print(dists_gmm)
print(dists_pmp)
print(np.mean(dists_gmm))
print(np.mean(dists_pmp))