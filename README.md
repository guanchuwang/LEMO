## LEMO: Learning Shapley Manifold for Faithful Explanation

### Research Motivation

Shapley value's exponentially growing complexity hinders its use in real-world scenarios.
Although existing works introduce DNN-explainers to efficiently estimate Shapley values, the training relies on noisy proxy labels that badly affect the performance of trained explainers. 
It is challenging to alleviate the harm of the noise caused by proxy labels, due to its heterogeneity across different instances and features.
To solve this problem, we conduct an in-depth analysis on the proxy label, and verify that it satisfies the Gaussian distribution centered on the Shapley value, termed as **Shapley Manifold** in this work.
Building upon this insight, we propose Learning Shapley Manifold (LEMO) to alleviate the harm of heterogeneous noise for more accurate Shapley value estimation.
 
 
### LEMO

LEMO designs a DNN-explainer with double-head outputs towards estimating the mean and standard deviation, respectively;
and to aligns the explanation with the Shapley manifold by minimizing their Kullback-Leibler divergence, for each feature and instance. 
In this way, it alleviates the harm of heterogeneous noise by adaptively assembling the contribution of each proxy label by its quality (i.e., noise power), where **low noise power of proxy label, high importance to the training**.

### Dependency
````angular2html
torch 
numpy
matplotlib
seaborn
````

### Leverage our Pre-trained Explainer

Run our pre-trained explainer to reproduce our experiment results
````angular2html
python3 l2_plot.py
python3 throughput_plot.py
````

The L2 error compared with the true Shapley values and Throughput will be:
<div align=center>
<img width="240" height="200" src="https://anonymous.4open.science/r/LEMO-3604/figure/L2_vs_n_sample_adult.png">
<img width="200" height="200" src="https://anonymous.4open.science/r/LEMO-3604/figure/Throughput-adult.png">
</div>


