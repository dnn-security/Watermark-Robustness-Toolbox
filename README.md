## Watermark-Robustness-Toolbox - Official PyTorch Implementation
[![contact](https://img.shields.io/badge/contact-nlukas@uwaterloo.ca-yellow)](mailto:rbp5354@psu.edu)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![PyTorch 1.3.1](https://img.shields.io/badge/torch-1.3.1-green.svg?style=plastic)
![cuDNN 10.1.2](https://img.shields.io/badge/cudnn-10.1.2-green.svg?style=plastic)
[![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://crysp.uwaterloo.ca/research/mlsec/wrt)
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)

This repository contains the official PyTorch implementation of the following paper to appear at IEEE Security and Privacy 2022:

> **SoK: How Robust is Deep Neural Network Image Classification Watermarking?**<br>
> Nils Lukas, Edward Jiang, 
> Xinda Li, Florian Kerschbaum<br>
> [insert arxiv link]
>
> **Abstract:** *Deep Neural Network (DNN) watermarking is a method for provenance verification of DNN models. Watermarking should be robust against watermark removal attacks that derive a surrogate model that evades provenance verification. Many watermarking schemes that claim robustness have been proposed, but their robustness is only validated in isolation against a relatively small set of attacks. There is no systematic, empirical evaluation of these claims against a common, comprehensive set of removal attacks. This uncertainty about a watermarking scheme's robustness causes difficulty to trust their deployment in practice. In this paper, we evaluate whether recently proposed watermarking schemes that claim robustness are robust against a large set of removal attacks. We survey methods from the literature that (i) are known removal attacks, (ii) derive surrogate models but have not been evaluated as removal attacks, and (iii) novel removal attacks. Weight shifting, transfer learning and smooth retraining are novel removal attacks adapted to the DNN watermarking schemes surveyed in this paper. We propose taxonomies for watermarking schemes and removal attacks. Our empirical evaluation includes an ablation study over sets of parameters for each attack and watermarking scheme on the image classification datasets CIFAR-10 and ImageNet. Surprisingly, our study shows that none of the surveyed watermarking schemes is robust in practice. We find that schemes fail to withstand adaptive attacks and known methods for deriving surrogate models that have not been evaluated as removal attacks. This points to intrinsic flaws in how robustness is currently evaluated. Our evaluation includes a discussion of the runtime of each attack to underpin their practical relevance. While none of the schemes is robust against all attacks, none of the attacks removes all watermarks. We show that attacks can be combined and find combined attacks that remove all watermarks. We show that watermarking schemes need to be evaluated against a more extensive set of removal attacks with a more realistic adversary model. Our source code and a complete dataset of evaluation results will be made publicly available, which allows to independently verify our conclusions.*

## Features

All watermarking schemes and removal attacks are configured for the image classification datasets 
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (32x32 pixels, 10 classes) and [ImageNet](https://www.image-net.org) (224x224 pixels, 1k classes). 
We implemented the following *watermarking schemes*, sorted by their categories:

- **Model Independent**:
[Adi](https://www.usenix.org/conference/usenixsecurity18/presentation/adi),
  [Content](https://dl.acm.org/doi/abs/10.1145/3196494.3196550?casa_token=RZrfzSIO_uwAAAAA:N7ohyz15GCGfoXRMtew-dX5dV-heZyI-N5Tod1xyKFWb46MXLPeqdfhMLizAFXlVE_VfZP_m2T3M), 
  [Noise](https://dl.acm.org/doi/abs/10.1145/3196494.3196550?casa_token=RZrfzSIO_uwAAAAA:N7ohyz15GCGfoXRMtew-dX5dV-heZyI-N5Tod1xyKFWb46MXLPeqdfhMLizAFXlVE_VfZP_m2T3M),
  [Unrelated](https://dl.acm.org/doi/abs/10.1145/3196494.3196550?casa_token=RZrfzSIO_uwAAAAA:N7ohyz15GCGfoXRMtew-dX5dV-heZyI-N5Tod1xyKFWb46MXLPeqdfhMLizAFXlVE_VfZP_m2T3M)
- **Model Dependent**:
[Jia](https://www.usenix.org/conference/usenixsecurity21/presentation/jia), 
  [Frontier Stitching](https://link.springer.com/article/10.1007/s00521-019-04434-z),
  [Blackmarks](https://arxiv.org/abs/1904.00344)
- **Parameter Encoding**: 
[Uchida](https://dl.acm.org/doi/abs/10.1145/3078971.3078974?casa_token=H5HTBeo2JDAAAAAA:P5P93MufED9DZZ5zAfqaaIJ5x2Y81t-HKfQLVPsRTC7XSaN7NaWUZA-1Wg2_F0ROIFCXzapYjsFs)
  , [DeepSigns](https://dl.acm.org/doi/abs/10.1145/3297858.3304051),
  [DeepMarks](https://dl.acm.org/doi/abs/10.1145/3323873.3325042)
- **Active**: [DAWN](https://arxiv.org/abs/1906.00830)

.. and the following *removal attacks*, sorted by their categories:

- **Input Preprocessing**:
[Input Reconstruction](https://arxiv.org/abs/1911.10291),
  [JPEG Compression](https://arxiv.org/abs/1608.00853),
  [Input Quantization](https://arxiv.org/abs/1904.08444),
  [Input Smoothing](https://arxiv.org/abs/1704.01155),
  [Input Noising](https://arxiv.org/abs/1707.06728),
Input Flipping, 
  [Feature Squeezing](https://arxiv.org/abs/1704.01155)

- **Model Modification**:
[Adversarial Training](https://arxiv.org/abs/1706.06083),
  [Fine-Tuning (FTLL, FTAL, RTAL, RTLL)](https://dl.acm.org/doi/abs/10.1145/3078971.3078974?casa_token=H5HTBeo2JDAAAAAA:P5P93MufED9DZZ5zAfqaaIJ5x2Y81t-HKfQLVPsRTC7XSaN7NaWUZA-1Wg2_F0ROIFCXzapYjsFs),
  [Weight Quantization](https://arxiv.org/abs/1609.07061), 
  [Label Smoothing](https://arxiv.org/abs/1512.00567),
  [Fine Pruning](https://arxiv.org/abs/1805.12185),
  Feature Permutation, 
  [Weight Pruning](https://arxiv.org/abs/1710.01878),
  Weight Shifting,
  [Neural Cleanse](https://ieeexplore.ieee.org/document/8835365), 
  [Regularization](https://arxiv.org/abs/1906.07745),
  [Neural Laundering](https://arxiv.org/abs/2004.11368), 
  [Overwriting](https://dl.acm.org/doi/abs/10.1145/3078971.3078974?casa_token=H5HTBeo2JDAAAAAA:P5P93MufED9DZZ5zAfqaaIJ5x2Y81t-HKfQLVPsRTC7XSaN7NaWUZA-1Wg2_F0ROIFCXzapYjsFs)
- **Model Extraction**:
[Knockoff Nets (Random Selection)](https://arxiv.org/abs/1812.02766),
  [Distillation](https://arxiv.org/abs/1503.02531), 
  Transfer Learning, 
  [Retraining](https://arxiv.org/abs/1609.02943),
  Smooth Retraining, 
  Cross-Architecture Retraining
  
## Get Started
At this point, the Watermark-Robustness-Toolbox project is not available as a 
standalone pip package, but we are working on allowing an installation via pip. 
We describe a manual installation and usage. 
First, install all dependencies via pip.
```shell
$ pip install -r requirements.txt
```

The following four main scripts provide the entire toolbox's functionality:

- *train.py*: Pre-trains an unmarked neural network. 
- *embed.py*: Embeds a watermark into a pre-trained neural network. 
- *steal.py*: Performs a removal attack against a watermarked neural network.
- *decision_threshold.py*: Computes the decision threshold for a watermarking scheme. 

We use the [mlconfig](https://github.com/narumiruna/mlconfig) library to pass configuration hyperparameters to each script. 
Configuration files used in our paper for CIFAR-10 and ImageNet can be found in the ``configs/`` directory. 
Configuration files store **all hyperparameters** needed to reproduce an experiment. 
### Step 1: Pre-train a Model on CIFAR-10
```shell
$ python train.py --config configs/cifar10/train_configs/resnet.yaml
```
This creates an ``outputs`` directory and saves a model file at ``outputs/cifar10/null_models/resnet/``.

### Step 2: Embed an Adi Watermark
```shell
$ python embed.py --wm_config configs/cifar10/wm_configs/adi.yaml \
                  --filename outputs/cifar10/null_models/resnet/best.pth
```
This embeds an Adi watermark into the pre-trained model from 'Example 1' and saves (i) the watermarked model and
(ii) all data to read the watermark under ``outputs/cifar10/wm/adi/00000_adi/``. 

### Step 3: Attempt to Remove a Watermark
```shell
$ python steal.py --attack_config configs/cifar10/attack_configs/ftal.yaml \
                  --wm_dir outputs/cifar10/wm/adi/00000_adi/
```
This runs the Fine-Tuning (FTAL) removal attack against the watermarked model and creates a surrogate model stored under
``outputs/cifar10/attacks/ftal/``. The directory also contains human-readable debug files, such as the surrogate model's watermark and 
test accuracies. 

## Datasets
Our toolbox currently implements custom data loaders (*class WRTDataLoader*) for the following datasets. 

- CIFAR-10
- ImageNet (needs manual download)
- Omniglot (needs manual download)
- Open Images (needs manual download)

## Documentation
We are actively working on documenting the parameters of each watermarking scheme and removal attack. 
At this point, we can only refer to the method's source code (at ``wrt/defenses/`` and ``wrt/attacks/``).
Soon we will host a complete documentation of all parameters, so stay tuned!

## Contribute
We encourage authors of watermarking schemes or removal attacks to implement their methods in the Watermark-Robustness-Toolbox 
to make them publicly accessible in a unified framework. 
Our aim is to improve reproducibility which makes it easier to evaluate a scheme's robustness. 
Any contributions or suggestions for improvements are welcome and greatly appreciated.
This toolbox is maintained as part of a university project by graduate students. 

## Reference
The codebase has been based off an early version of the 
[Adversarial-Robustness-Tooblox](https://github.com/Trusted-AI/adversarial-robustness-toolbox).

## Cite our paper
```
@InProceedings{lukas2022watermarkingsok,
  title={SoK: How Robust is Deep Neural Network Image Classification Watermarking?}, 
  author={Lukas, Nils and Jiang, Edward and Li, Xinda and Kerschbaum, Florian},
  year={2022},
  booktitle={IEEE Symposium on Security and Privacy}
}
```




