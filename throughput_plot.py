
import torch
import matplotlib.pyplot as plt
import numpy as np

import sys, os
sys.path.append("../")
from utils import val_prepare, val
import seaborn as sns

checkpoint_buf = {

    "Census Income":{
            "KS":[
                "output/Adult_KS_s_256.pth.tar",
            ],
            "KS-Pair":[
                "output/Adult_KS-Pair_s_64.pth.tar",
            ],
            "PS":[
                "output/Adult_PS_s_16.pth.tar",
            ],
            "PS-Ant*":[
                "output/Adult_PS-Antithetic_s_4.pth.tar",
            ],
            "GradShap": [
                "output/Adult_Gradshap_s_0.pth.tar",
            ],

            "checkpoint_target_model_softmax": torch.load("../../exp_adult/adult_model/model_adult_m_1_r_0-softmax-sv.pth.tar"),
            "checkpoint_lsm": torch.load("output/model_adult_softplus_r_0_explainer_bs_2_sample_4_layer_5_hidden_256_lr_0.0002_seed_7-softmax.pth.tar"),
            "checkpoint_fastshap": torch.load("output/fastshap_adult_bs_2_sample_4_layer_5_hidden_256_lr_0.0002_seed_7-softmax.pth.tar"),
    },

}


algorithm_buf = ["KS", "KS-Pair", "PS", "PS-Ant*", "GradShap"] # "LFS",
marker_buf = ['^', '>', '<', 'v', 's', 'o', 'p', 'h']
color_buf = ["red", "green", "magenta", "black", "#a87900", "cyan", "blue", "slateblue"]

algorithm_throughput = {"KS": [],
                        "KS-Pair": [],
                        "PS": [],
                        "PS-Ant*": [],
                        "GradShap": [],
                        "FastSHAP": [],
                        "LEMO": []
                        }

dataset_buf = ["Census Income"]
dataset_short_buf = ["Income"]

for dataset_idx, dataset in enumerate(dataset_buf):

    for alg_index, alg in enumerate(algorithm_buf):
        checkpoint_fname = checkpoint_buf[dataset][alg][0]
        checkpoint = torch.load(checkpoint_fname)
        time_cost = checkpoint["total_time"]
        Throughput = 1/time_cost
        algorithm_throughput[alg].append(Throughput)

        print(alg, Throughput)

    ######### FastSHAP
    target_model, explainer, test_loader, reference = val_prepare(checkpoint_buf[dataset]["checkpoint_target_model_softmax"],
                                                                  checkpoint_buf[dataset]["checkpoint_fastshap"], batch_size=256)
    l2_error_lsm, l2_error_avg, l2_error_std, time_fastshap = val(test_loader, explainer, target_model.forward_softmax, reference)
    Throughput_fastshap = 1 / time_fastshap
    algorithm_throughput["FastSHAP"].append(Throughput_fastshap)
    print("FastSHAP", Throughput_fastshap)


    ######### LEMO
    target_model, explainer, test_loader, reference = val_prepare(checkpoint_buf[dataset]["checkpoint_target_model_softmax"],
                                                                  checkpoint_buf[dataset]["checkpoint_lsm"], batch_size=256)
    l2_error_lsm, l2_error_avg, l2_error_std, time_lsm = val(test_loader, explainer, target_model.forward_softmax, reference)
    Throughput_lsm = 1 / time_lsm
    algorithm_throughput["LEMO"].append(Throughput_lsm)
    print("LEMO", Throughput_lsm)

print(algorithm_throughput)
bar_width = 5
plt.figure(figsize=((5, 5)))

for index, method in enumerate(algorithm_throughput.keys()):
    bar_x_axis = np.arange(len(algorithm_throughput[method])) * 16 * bar_width + bar_width * index + bar_width
    plt.bar(bar_x_axis, np.log10(np.array(algorithm_throughput[method])), width=bar_width, color=color_buf[index], label=method)

plt.grid(axis='y')
plt.yticks(np.arange(6), ["$10^{%s}$" % x for x in np.arange(6)], fontsize=18)
plt.ylabel("Throughput", fontsize=18)
h_leg = plt.legend(fontsize=17.5, frameon=True,
                   ncol=1, loc="upper right")

plt.ylim([0, 5.5])
plt.yticks(fontsize=18)
plt.xticks(bar_x_axis - 3*bar_width, dataset_short_buf, fontsize=18)
for leg_line in h_leg.get_lines():
    leg_line.set_linewidth(6)
plt.xlim([-4*bar_width, bar_width*32])
plt.subplots_adjust(left=0.17, bottom=0.07, top=0.99, right=0.99, wspace=0.01)
plt.savefig("figure/Throughput-adult.png")
# plt.show()
plt.close()

