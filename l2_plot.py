
import torch
import matplotlib.pyplot as plt
import numpy as np

import sys, os
from utils import val_prepare, val
import seaborn as sns

checkpoint_buf = {

    "LEMO":[
        "output/model_adult_softplus_r_0_explainer_bs_2_sample_4_epoch_200_layer_5_hidden_256_lr_0.0002_seed_7-softmax.pth.tar",
    ],
    "KernelShap":[
        "output/Adult_KS_s_4.pth.tar",
        "output/Adult_KS_s_8.pth.tar",
        "output/Adult_KS_s_16.pth.tar",
        "output/Adult_KS_s_32.pth.tar",
        "output/Adult_KS_s_64.pth.tar",
        "output/Adult_KS_s_128.pth.tar",
        "output/Adult_KS_s_256.pth.tar",
        "output/Adult_KS_s_512.pth.tar",
        "output/Adult_KS_s_1024.pth.tar",
        "output/Adult_KS_s_2048.pth.tar",
        "output/Adult_KS_s_4096.pth.tar",
    ],
    "KS-Pair":[
        "output/Adult_KS-Pair_s_4.pth.tar",
        "output/Adult_KS-Pair_s_8.pth.tar",
        "output/Adult_KS-Pair_s_16.pth.tar",
        "output/Adult_KS-Pair_s_32.pth.tar",
        "output/Adult_KS-Pair_s_64.pth.tar",
        "output/Adult_KS-Pair_s_128.pth.tar",
        "output/Adult_KS-Pair_s_256.pth.tar",
        "output/Adult_KS-Pair_s_512.pth.tar",
        "output/Adult_KS-Pair_s_1024.pth.tar",
        "output/Adult_KS-Pair_s_2048.pth.tar",
    ],
    "PS":[
        "output/Adult_PS_s_1.pth.tar",
        "output/Adult_PS_s_2.pth.tar",
        "output/Adult_PS_s_4.pth.tar",
        "output/Adult_PS_s_8.pth.tar",
        "output/Adult_PS_s_16.pth.tar",
        "output/Adult_PS_s_32.pth.tar",
        "output/Adult_PS_s_64.pth.tar",
        "output/Adult_PS_s_128.pth.tar",
        "output/Adult_PS_s_256.pth.tar",
        "output/Adult_PS_s_512.pth.tar",
        "output/Adult_PS_s_1024.pth.tar",
        "output/Adult_PS_s_2048.pth.tar",
    ],
    "PS-Antithetic":[
        "output/Adult_PS_s_1.pth.tar",
        "output/Adult_PS-Antithetic_s_4.pth.tar",
        "output/Adult_PS-Antithetic_s_8.pth.tar",
        "output/Adult_PS-Antithetic_s_16.pth.tar",
        "output/Adult_PS-Antithetic_s_32.pth.tar",
        "output/Adult_PS-Antithetic_s_64.pth.tar",
        "output/Adult_PS-Antithetic_s_128.pth.tar",
        "output/Adult_PS-Antithetic_s_256.pth.tar",
        "output/Adult_PS-Antithetic_s_512.pth.tar",
        "output/Adult_PS-Antithetic_s_1024.pth.tar",
    ],
}

checkpoint_target_model = torch.load("model/model_adult_m_1_r_0-sv.pth.tar")
checkpoint_target_model_softmax = torch.load("model/model_adult_m_1_r_0-softmax-sv.pth.tar")
checkpoint_lsm = torch.load("output/model_adult_softplus_r_0_explainer_bs_2_sample_4_layer_5_hidden_256_lr_0.0002_seed_7-softmax.pth.tar")
checkpoint_fastshap = torch.load("output/fastshap_adult_bs_2_sample_4_layer_5_hidden_256_lr_0.0002_seed_7-softmax.pth.tar")


algorithm_buf = ["KernelShap", "KS-Pair", "PS", "PS-Antithetic"] # "LFS",
marker_buf = ['^', '>', '<', 'v', 's', 'o', 'p', 'h']
color_buf = ["blue", "#a87900", "magenta", "green", "red", "orange", "black", "#5d1451"]

L2_buf_buf = []
L2_std_buf_buf = []
L2_sample_buf_buf = []
n_sample_buf_buf = []
time_buf_buf = []

for alg_index, alg in enumerate(algorithm_buf):

    fname_buf = checkpoint_buf[alg]
    L2_buf = np.zeros((len(fname_buf),))
    L2_std_buf = np.zeros((len(fname_buf),))
    n_sample_buf = np.zeros((len(fname_buf),))
    time_buf = np.zeros((len(fname_buf),))
    L2_error_buf = np.zeros((len(fname_buf),))
    L2_sample_buf = []
    n_sample_buf = []

    for checkpoint_index, checkpoint_fname in enumerate(fname_buf):
        checkpoint = torch.load(checkpoint_fname)
        n_sample = checkpoint["sample_num"]
        shapley_value = checkpoint["attr"]

        # print(total_time)

        N = shapley_value.shape[0]
        feature_num = shapley_value.shape[-1]
        shapley_value_gt = checkpoint_target_model["test_shapley_value"][0:N]

        # absolute_error = torch.abs(shapley_value - shapley_value_gt).sum(dim=1)
        l2_error = torch.sqrt(torch.square(shapley_value - shapley_value_gt).sum(dim=-1))

        l2_error[torch.isnan(l2_error)] = 0
        l2_error[l2_error > 10] = 0
        L2_sample_buf.append(l2_error.unsqueeze(dim=0))
        n_sample_buf.append(n_sample)

    L2_sample_buf = torch.cat(L2_sample_buf, dim=0)

    print(alg)
    print(n_sample_buf)
    print(L2_sample_buf.T.mean(dim=0))

    if alg == "PS" or alg == "PS-Antithetic":
        multi_factor = checkpoint_target_model["reference_test"].shape[0]
    else:
        multi_factor = 1

    sns.tsplot(time=np.log2(np.array(n_sample_buf) * multi_factor), data=L2_sample_buf.T,
               # marker=marker_buf[alg_index],
               condition=algorithm_buf[alg_index],
               linewidth=1, markersize=8,
               color=color_buf[alg_index],
               ci=[100]
               )

######### GradSHAP
checkpoint_gradshap = torch.load("output/Adult_Gradshap_s_0.pth.tar")
shapley_value_gradshap = checkpoint_gradshap["attr"]
shapley_value_gt = checkpoint_target_model_softmax["test_shapley_value"]
l2_error_gradshap = torch.sqrt(torch.square(shapley_value_gradshap - shapley_value_gt).sum(dim=-1)).mean()
print("GradSHAP", l2_error_gradshap)
n_sample_buf = [2**x for x in range(15)]
plt.plot(np.log2(n_sample_buf), [l2_error_gradshap] * len(n_sample_buf), color="black", linewidth=2, label="GradSHAP")

######### FastSHAP
target_model, explainer, test_loader, reference = val_prepare(checkpoint_target_model_softmax, checkpoint_fastshap)
l2_error_lsm, l2_error_avg, l2_error_std, _ = val(test_loader, explainer, target_model.forward_softmax, reference)
print("FastSHAP", l2_error_lsm, l2_error_avg, l2_error_std)
plt.plot(np.log2(n_sample_buf), [l2_error_avg] * len(n_sample_buf), color="purple", linewidth=2, label="FastSHAP")


######### LEMO
target_model, explainer, test_loader, reference = val_prepare(checkpoint_target_model_softmax, checkpoint_lsm)
l2_error_lsm, l2_error_avg, l2_error_std, _ = val(test_loader, explainer, target_model.forward_softmax, reference)
print("LEMO", l2_error_lsm, l2_error_avg, l2_error_std)
plt.plot(np.log2(n_sample_buf), [l2_error_avg] * len(n_sample_buf), color="red", linewidth=2, label="LEMO")

plt.xlabel("Eval. number", fontsize=18)
plt.ylabel("$L_2$ Error", fontsize=18)
plt.legend(loc='upper right', fontsize=18, frameon=True)

plt.xticks(np.log2(n_sample_buf), ["$2^{%s}$" % int(x) for x in np.log2(n_sample_buf)], fontsize=18)
# plt.gca().ticklabel_format(style='sci', scilimits=(0, -3), axis='y')

# plt.xticks(fontsize=15)
plt.yticks(fontsize=18)
plt.xlim([3.8, 12])
plt.ylim([0, 0.5])
# plt.grid(axis="y")
plt.subplots_adjust(left=0.14, bottom=0.14, top=0.97, right=0.96, wspace=0.01)
plt.savefig("figure/L2_vs_n_sample_adult.png")
# plt.show()
plt.close()

