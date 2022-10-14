# U-MAE

This repo includes the code for Neurips 2022 paper [How Mask Matters: Towards Theoretical Understandings of Masked Autoencoders.]

## Running Instructions
This repo is based on the official code of MAE (https://github.com/facebookresearch/mae).  And the running instructions can be found in ``OFFICIAL_MAE-README.md``.

## The Uniformity-enhanced MAE Loss

The main differences between our and the official code include:

1. We add a new uniformity regularizier, which is implemented in ``uni_reg.py``. Our implementation of the regularizer is based on the spectral contrasive loss (https://arxiv.org/abs/2106.04156).
2. We add two hyperparemeters in ``pretrain.sh``.  ``beta`` is the coffecient of the regularizer and ``tau`` is the coffecient to normalize the similarity matrix. Both ``beta`` and ``tau`` are designed to control the strength of the regularizier (default ``tau=0.1`` and ``beta = 0.0001``).
3. We add a linear classfier to monitor the online linear accuracy whose gradient will not be backward propagated to the backbone.
4. We follow most of the settings in ``pretrain.sh`` of the official MAE code. Except we set the epoch to 200 and the corresponding warmup epoch to 20.

## Acknowledgement
Our code is follow the default settings of the official implementation of MAE (https://github.com/facebookresearch/mae).
