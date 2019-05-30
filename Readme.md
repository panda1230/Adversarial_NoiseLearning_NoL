# Official PyTorch implementation for the ICML 2019 Workshop paper: 
## "Implicit Generative Modeling of Random Noise during training improves Adversarial Robustness" https://arxiv.org/abs/1807.02188

#### ICML 2019 Workshop on Uncertainty & Robustness in Deep Learning (https://sites.google.com/view/udlworkshop2019/home)
#### Acronym definition: NoL- Noise-based prior learning (our proposal), SGD- Stochastic Gradient Descent
This software allows users to reproduce the results from the paper
including 
1) Principal Component Analysis- Variance and Cosine Distance Results comparing adversarial vs. clean input response of a model, 
2) Adversarial Accuracy- Projected Gradient Descent (or PGD) attack, FGSM attack results,
3) Loss Surface visualization for blackbox and whitebox attack for SGD vs. NoL
4) Adversarial Subspace dimensionality measured with the GAAS method[https://arxiv.org/pdf/1704.03453.pdf]. 

The ipython notebooks provide a good comparison between SGD vs. NoL training scenario for ResNet18 model trained on CIFAR10 dataset.

The 'checkpoint_submission' folder contains the saved files:
1) Adv_space_NoL_epsilon_8.h5 has the saved PC projections for the adversarial/clean input visualization in PC subspace for NoL trained model.
2) Adv_space_SGD_epsilon_8.h5 has the saved PC projections for the adversarial/clean input visualization in PC subspace for SGD trained model.
3) PC_dim_NoL.h5 (or PC_dim_SGD.h5) has the PC projections for the adversarial/clean input to plot cosine distance for NoL (or SGD) model.

The 'checkpoint_submission' folder also contains trained model files:
1) No_noise_ckpt5_lrstep.h5: Target model to be attacked (trained with SGD) 
2) noise_v1.h5, state_with_noise_v1.h5 (has the learnt noise templates): Target model to be attacked (trained with NoL) 
3) No_noise_ckpt4_lrstep.h5: Source model to create blackbox attacks (trained with SGD) 

- Ipython notebooks for SGD training scenario <Cifar10_ResNet18_SGD_submission.ipynb> and NoL training scenario <Cifar10_ResNet18_NoL_submission.ipynb> is present. 
- <Adversarial_Dimensionality_NoLvsSGD_submission.ipynb> plots the adversarial dimensionality calculated using GAAS and compares the dimension of NoL vs. SGD.
- <PC_CosineDistance_NoLvsSGD_submission.ipynb> plots the cosine distance measured for PC projection of Conv1 layer of a model in response to clan and adversarial inputs and compares the distance for NoL vs. SGD.

Notebooks have detailed comments that will help the users follow each step. For any other queries, please contact me @the corresponding author email id mentioned in the paper(https://arxiv.org/abs/1807.02188).

 | Dependencies  |
| ------------- |
| python == 3.6     |
| pytorch == 0.4.1     |
| cuda92|
| torchvision|
| matplotlib|
| scikit-learn|
|scipy        |

### Please consider citing our paper if you find this work useful for your research.


```
 @article{panda2018explainable,
  title={Explainable learning: Implicit generative modelling during training for adversarial robustness},
  author={Panda, Priyadarshini and Roy, Kaushik},
  journal={arXiv preprint arXiv:1807.02188v3},
  year={2018}
}
```
