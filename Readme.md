# Official PyTorch implementation for the ICML 2019 Workshop paper: "Implicit Generative Modeling of Random Noise during training improves Adversarial Robustness" https://arxiv.org/abs/1807.02188

The 'checkpoint_submission' folder contains already trained model files. 

Ipython notebooks for SGD training scenario and NoL training scenario is present. 
Notebooks have detailed comments that will help the users follow each step. 
For any other queries, please contact me @the corresponding author email id mentioned in the paper(https://arxiv.org/abs/1807.02188).

This software allows users to reproduce the results from the paper
including Principal Component Analysis- Variance and Cosine Distance Results and Adversarial Accuracy- PGD attack, FGSM attack results. 

The ipython notebooks provide a good comparison between SGD vs. NoL training scenario for ResNet18 model trained on CIFAR10 dataset.

 | Dependencies  |
| ------------- |
| python == 3.6     |
| pytorch == 0.4.1     |
| cuda92|
| torchvision|
| matplotlib|
| scikit-learn|
|scipy        |

### Please consider citing the paper if you find this work useful for your research.


```
 @article{panda2018explainable,
  title={Explainable learning: Implicit generative modelling during training for adversarial robustness},
  author={Panda, Priyadarshini and Roy, Kaushik},
  journal={arXiv preprint arXiv:1807.02188v3},
  year={2018}
}
```
