## LETA: Learning Transferable Attribution for Generic Vision Explainer

### Research Motivation

Shapley value's exponentially growing complexity hinders its use in real-world scenarios.
Although existing works introduce DNN-explainers to efficiently estimate Shapley values, the training relies on noisy proxy labels that badly affect the performance of trained explainers. 
It is challenging to alleviate the harm of the noise caused by proxy labels, due to its heterogeneity across different instances and features.
To solve this problem, we conduct an in-depth analysis on the proxy label, and verify that it satisfies the Gaussian distribution centered on the Shapley value, termed as **Shapley Manifold** in this work.
Building upon this insight, we propose Learning Shapley Manifold (LEMO) towards minimizing the Kullback-Leibler divergence aligned with the Shapley manifold for each instance and feature. 
In this way, it alleviates~the harm of heterogeneous noise by adaptively assembling the contribution of each proxy label by the noise power, where **low noise power of proxy label, high importance to the training**.


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

The heatmap of local explanation will be
<div align=center>
<img width="180" height="150" src="https://anonymous.4open.science/r/LEMO-3604/figure/L2_vs_n_sample_adult.png">
<img width="120" height="150" src="https://anonymous.4open.science/r/LEMO-3604/figure/Throughput-adult.png">
</div>


