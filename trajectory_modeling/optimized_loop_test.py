import time
import numpy as np
from TP_PMP import utils

error_threshold = 0.00001
test_rounds = 1

y_dim = 28
w_dim = 350

duration = 50
demos = 5
n_components = 4

Y = np.array([[np.random.random(y_dim) for i in range(duration)] for d in range(demos)])
Phi = np.array([[np.random.random((y_dim, w_dim)) for i in range(duration)] for d in range(demos)])
inv_sig_w = np.array([np.eye(w_dim) for c in range(n_components)])
inv_sig_y = np.array([np.eye(y_dim) for c in range(n_components)])
mu_w = np.array([np.random.random(w_dim) for c in range(n_components)])
times = [duration for i in range(demos)]
temp_alpha = np.random.sample(size=n_components)
alpha_global = temp_alpha / sum(temp_alpha)


def orig_code():
    w_means = []
    w_covs = []
    alpha = alpha_global
    for n, time in enumerate(times):
        Tn = len(Y[n])
        wns = []
        Swns = []
        for i in range(n_components):

            sum_mean = np.dot(inv_sig_w[i], mu_w[i])
            sum_cov = inv_sig_w[i]
            for t in range(Tn):
                phi_nt = Phi[n][t]
                tmp1 = np.dot(np.transpose(phi_nt), inv_sig_y[i])
                sum_mean = sum_mean + (tmp1 @ Y[n][t])
                sum_cov = sum_cov + np.dot(tmp1, phi_nt)
            Swn = utils.force_sym(np.linalg.inv(sum_cov))
            wn = np.dot(Swn, sum_mean)
            wns.append(wn)

            Swns.append(Swn)
        w_means.append(np.dot(alpha[np.newaxis, :], wns).flatten())
        w_covs.append(np.einsum('ij,jkl->kl', alpha.reshape(1, -1), Swns))
    return w_means, w_covs


def modified_code():
    w_means = []
    w_covs = []
    alpha = alpha_global
    for n, time in enumerate(times):
        Tn = len(Y[n])
        wns = []
        Swns = []
        for i in range(n_components):

            sum_mean = np.dot(inv_sig_w[i], mu_w[i])
            sum_cov = inv_sig_w[i]
            concat_Phi = np.concatenate(Phi[n], axis=1)
            # concat_Y = np.concatenate(Y[n])
            concat_tmp1 = concat_Phi.transpose() @ inv_sig_y[i]
            concat_tmp1 = concat_tmp1.reshape(Tn, w_dim, -1)
            for t in range(Tn):
                sum_mean = sum_mean + concat_tmp1[t] @ Y[n][t]
                sum_cov = sum_cov + concat_tmp1[t] @ Phi[n][t]
            Swn = utils.force_sym(np.linalg.inv(sum_cov))
            wn = np.dot(Swn, sum_mean)
            wns.append(wn)

            Swns.append(Swn)
        w_means.append(np.dot(alpha[np.newaxis, :], wns).flatten())
        w_covs.append(np.einsum('ij,jkl->kl', alpha.reshape(1, -1), Swns))
    return w_means, w_covs

def einsum_code_v1():
    w_means = []
    w_covs = []
    alpha = alpha_global
    inv_time = 0
    for n, _ in enumerate(times):
        # Assume the duration is equal for all demos
        Tn = duration
        wns = []
        Swns = []
        for i in range(n_components):
            concat_tmp1 = np.einsum('gji,jk', Phi[n], inv_sig_y[i], optimize=True)
            sum_mean = np.dot(inv_sig_w[i], mu_w[i]) + np.einsum("ijk,ik", concat_tmp1, Y[n], optimize=True)

            sum_cov = inv_sig_w[i] + np.einsum("ijk,ikm", concat_tmp1, Phi[n], optimize=True)
            Swn = utils.force_sym(np.linalg.inv(sum_cov))

            wn = np.dot(Swn, sum_mean)
            wns.append(wn)
            Swns.append(Swn)
        w_means.append(np.dot(alpha[np.newaxis, :], wns).flatten())
        w_covs.append(np.einsum('ij,jkl->kl', alpha.reshape(1, -1), Swns, optimize=True))
    return w_means, w_covs


def einsum_code_v2():
    wns = []
    Swns = []
    alpha = alpha_global
    inv_time = 0
    for i in range(n_components):
        subcode_time = time.time()
        inv_sig_mu_w_prod =  inv_sig_w[i] @ mu_w[i]
        concat_tmp1 = np.einsum("gikj,kl", Phi, inv_sig_y[i], optimize=True)

        sum_mean_all = inv_sig_mu_w_prod[np.newaxis,:] + \
                       np.matmul(concat_tmp1, np.expand_dims(Y, 2).transpose([0,1,3,2])).sum(axis=1).squeeze(-1)
        inv_time += time.time() - subcode_time
        temps = []
        for n in range(demos):
            temps.append(np.einsum("ijk,ikl", concat_tmp1[n], Phi[n], optimize=True))
            # temps.append(np.matmul(concat_tmp1[n], Phi[n]).sum(0))
        temp = np.array(temps)
        sum_cov_all = inv_sig_w[i][np.newaxis,:].repeat(demos, 0) + temp\

        Swn = np.linalg.inv(sum_cov_all)

        # wn = np.einsum("gij,gj->gi", Swn, sum_mean_all, optimize=True)
        wn = np.matmul(Swn, np.expand_dims(sum_mean_all, 2)).squeeze(-1)
        wns.append(wn)
        Swns.append(Swn)
    wns = np.array(wns).swapaxes(1, 0)
    Swns = np.array(Swns).swapaxes(1, 0)
    w_means = [np.dot(alpha[np.newaxis, :], wns[n]).flatten() for n, _ in enumerate(times)]
    w_covs = [np.einsum('ij,jkl->kl', alpha.reshape(1, -1), Swns[n], optimize=True) for n, _ in enumerate(times)]
    return w_means, w_covs

def loop_free():
    alpha = alpha_global
    inv_sig_mu_w_prods = np.matmul(inv_sig_w, mu_w[:,:,np.newaxis]).squeeze(-1)

    # concat_tmp1 = np.matmul(Phi[:,np.newaxis,:,:,:].transpose([0,1,2,4,3]),
    #                         inv_sig_y[np.newaxis,:,np.newaxis,:,:])
    concat_tmp1 = np.einsum("ghij,kil->gkhjl", Phi, inv_sig_y, optimize=True)

    sum_mean_all = inv_sig_mu_w_prods[np.newaxis,:] + \
                   np.matmul(concat_tmp1, Y[:,np.newaxis,:,:,np.newaxis]).sum(axis=2).squeeze(-1)

    t_start = time.time()
    temp = np.einsum("ghijk,gikl->ghijl", concat_tmp1, Phi)
    # temp = concat_tmp1 @ Phi[:,np.newaxis,:,:,:]
    # temps = []
    # for i in range(demos):
    #     temps.append(concat_tmp1[i] @ Phi[i])
    # temp = np.array(temps)
    print(time.time() - t_start)
    sum_cov_all = inv_sig_w[np.newaxis,:,:,:] + temp.sum(2)

    Swn = np.linalg.inv(sum_cov_all)
    wn = np.matmul(Swn, sum_mean_all[:,:,:,np.newaxis]).squeeze(-1)
    w_means = [np.dot(alpha[np.newaxis, :], wn[n]).flatten() for n in range(wn.shape[0])]
    w_covs = [np.einsum('ij,jkl->kl', alpha.reshape(1, -1), Swn[n], optimize=True) for n in range(wn.shape[0])]

    # print(f"Subcode_matmul time cost: {inv_time}")
    return w_means, w_covs


# Original Code for validating output
start_time = time.time()
for _ in range(test_rounds):
    orig_result = orig_code()
print("Original --- %s seconds ---" % ((time.time() - start_time)/test_rounds))

# Using einsum in place of matmul
start_time = time.time()
for _ in range(test_rounds):
    einsum_result = einsum_code_v1()
print("Einsum --- %s seconds ---" % ((time.time() - start_time)/test_rounds))

assert (np.abs(np.array(einsum_result[0])-np.array(orig_result[0])) < error_threshold).all()
assert (np.abs(np.array(einsum_result[1])-np.array(orig_result[1])) < error_threshold).all()

# Light modification by concatenating the Phi first
start_time = time.time()
for _ in range(test_rounds):
    modified_result = einsum_code_v2()
print("Modified --- %s seconds ---" % ((time.time() - start_time)/test_rounds))

assert (np.abs(np.array(modified_result[0])-np.array(orig_result[0])) < error_threshold).all()
assert (np.abs(np.array(modified_result[1])-np.array(orig_result[1])) < error_threshold).all()


# Loop-free runtime and correctness
start_time = time.time()
for _ in range(test_rounds):
    loopfree_result = loop_free()
print("Loop-Free --- %s seconds ---" % ((time.time() - start_time)/test_rounds))

assert (np.abs(np.array(loopfree_result[0])-np.array(orig_result[0])) < error_threshold).all()
assert (np.abs(np.array(loopfree_result[1])-np.array(orig_result[1])) < error_threshold).all()

