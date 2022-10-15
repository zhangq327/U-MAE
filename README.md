# U-MAE

This repository includes the code for the Uniformity-enhanced Masked Autoencoder (**U-MAE**) proposed in the NeurIPS 2022 paper [How Mask Matters: Towards Theoretical Understandings of Masked Autoencoders](https://openreview.net/pdf?id=WOppMAJtvhv). 

U-MAE is an extension of [MAE (He et al., 2022)](https://arxiv.org/pdf/2111.06377.pdf) by further encouraging the feature uniformity of MAE. As shown below, U-MAE successfully addresses the dimensional collapse issue of MAE.

<p align="center">
  <img src="https://user-images.githubusercontent.com/16850758/195980285-48985231-fc68-40a1-b2d3-81462c5f868a.png" width="1000">
</p>


## Instructions
This repo is based on the [official code of MAE](https://github.com/facebookresearch/mae) with minor modifications below, and we follow all the default training and evaluation configurations of MAE. Please see their instructions [README_mae.md](README_mae.md) for details.

**Main differences.** In U-MAE, we introduce a ``uniformity_loss``  (implemented in ``loss_func.py``) as a uniformity regularization to the MAE loss. It has two  additional hyper-parameters that are included in ``pretrain.sh``:
* ``lamb`` (default to ``1e-4``) is the coefficient lambda of the uniformity regularizer in the U-MAE loss;
* ``tau`` (default to ``0.1``) is a temperature parameter to scale the feature similarity as adopted in SimCLR. 

**Minor points:**
1. We add a linear classifier to monitor the online linear accuracy and its gradient will not be backward propagated to the backbone encoder.
2. For efficiency, we only train U-MAE for 200 epochs, and accordingly, we adopt 20 warmup epochs.

## Acknowledgement

Our code follows the official implementations of MAE (https://github.com/facebookresearch/mae).
