import os
import numpy as np
import random
import TP_GMM.gmm as gmm
from test import get_bic, marginal_t, get_aic, get_position_difference_per_step
from matplotlib import pyplot as plt
from TP_PMP import pmp

data_dir = './TP_GMM/data/wiping'
root, dirs, files = next(os.walk(data_dir))
demos = []
for f in files:
    demo_file = os.path.join(root, f)
    temp = np.load(demo_file)
    demos.append(temp)

demos_with_t = []
for demo in demos:
    t = np.linspace(0,1, demo.shape[0])
    temp = np.c_[t, demo]
    demos_with_t.append(temp)

n_test = 20
dists_gmm, dists_pmp = [], []
inds = []
for i in range(n_test):
    demo_inds = list(range(len(demos_with_t)))
    n_train = 6
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
        print(j)
        data_gmm = np.concatenate(train_demos)
        t = np.linspace(0,1, train_demos[0].shape[0])
        n_states = j + 1
        n_data = data_gmm.shape[0]
        n_dims = data_gmm.shape[1]
        model_gmm = gmm.GMM(nb_states=n_states, nb_dim=n_dims)
        model_gmm.em(data_gmm, reg=1e-3, maxiter=200, verbose=True)
        LL = model_gmm.LL
        bic = get_bic(LL,  n_states * n_dims + n_states * (n_dims * (n_dims + 1) / 2), n_data)
        aic = get_aic(LL, n_states * n_dims + n_states * (n_dims * (n_dims + 1) / 2), n_data)
        gmms.append(model_gmm)
        bics.append(bic)
        aics.append(aic)
    # plt.plot(bics, label = 'bic')
    # plt.plot(aics, label = 'aic')
    # plt.legend()
    # plt.show()
    temp = []
    t = test_demo[:, 0]
    ground_truth = test_demo[:, 1:]
    for g in gmms:
        mu_gmm, sigma_gmm = marginal_t(g, t)
        average_dist = np.mean(get_position_difference_per_step(ground_truth, mu_gmm))
        temp.append(average_dist)
    # print(f'The distance for the best gmm model is : {np.min(temp)}, The best model is {np.argmin(temp)}')
    ind = np.argmin(temp)
    inds.append(ind)
    dist_gmm = np.min(temp)
    gmm_selected = gmms[ind]
    mu_gmm, sigma_gmm = marginal_t(gmm_selected, t)

    data_pmp = [demo[:, 1:] for demo in train_demos]
    times = [demo[:,0] for demo in train_demos]
    model_pmp = pmp.PMP(data_pmp, times, n_dims - 1 )
    model_pmp.train()
    mu_pmp, sigma_pmp = marginal_t(model_pmp, t)
    fig, axes = plt.subplots(3, 1)
    axes[0].plot(ground_truth[:, 0], ground_truth[:, 1])
    axes[0].set_title('Ground truth')
    axes[1].plot(mu_gmm[:, 0], mu_gmm[:, 1])
    axes[1].set_title('GMM prediction')
    axes[2].plot(mu_pmp[:, 0], mu_pmp[:, 1])
    axes[2].set_title('PMP prediction')
    plt.show()
    dist_pmp = np.mean(get_position_difference_per_step(ground_truth, mu_pmp))
    dists_gmm.append(dist_gmm)
    dists_pmp.append(dist_pmp)

print(np.mean(dists_gmm))
print(np.mean(dists_pmp))
print(inds)