# U-MAE (Uniformity-enhanced Masked Autoencoder)

This repository includes a PyTorch implementation of the NeurIPS 2022 paper [How Mask Matters: Towards Theoretical Understandings of Masked Autoencoders](https://arxiv.org/pdf/2210.08344.pdf) authored by Qi Zhang*, [Yifei Wang*](https://yifeiwang77.github.io/), and [Yisen Wang](https://yisenwang.github.io/).

U-MAE is an extension of [MAE (He et al., 2022)](https://arxiv.org/pdf/2111.06377.pdf) by further encouraging the feature uniformity of MAE. As shown below, U-MAE successfully addresses the dimensional feature collapse issue of MAE.

<p align="center">
  <img src="https://user-images.githubusercontent.com/16850758/195980285-48985231-fc68-40a1-b2d3-81462c5f868a.png" width="1000">
</p>


## Instructions
This repo is based on the [official code of MAE](https://github.com/facebookresearch/mae) with minor modifications below, and we follow all the default training and evaluation configurations of MAE. Please see their instructions [README_mae.md](README_mae.md) for details.

**Main differences.** In U-MAE, we introduce a ``uniformity_loss``  (implemented in ``loss_func.py``) as a uniformity regularization to the MAE loss. It also introduces an additional hyper-parameter ``lamb`` (default to ``1e-2``) in ``pretrain.sh``, which represents the coefficient of the uniformity regularization in the U-MAE loss. 

**Minor points:**
1. We add a linear classifier to monitor the online linear accuracy and its gradient will not be backpropagated to the backbone encoder.
2. For efficiency, we only train U-MAE for 200 epochs, and accordingly, we adopt 20 warmup epochs.

## Citing this work
If you find the work useful, please cite the accompanying paper:
```
@inproceedings{zhang2022how,
  title={How Mask Matters: Towards Theoretical Understandings of Masked Autoencoders},
  author={Zhang, Qi and Wang, Yifei and Wang, Yisen},
  booktitle={NeurIPS},
  year={2022}
}
```

## Acknowledgement

Our code follows the official implementations of MAE (https://github.com/facebookresearch/mae).
