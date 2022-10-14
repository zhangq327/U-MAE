# U-MAE

This repo includes the code for Neurips 2022 paper [How Mask Matters: Towards Theoretical Understandings of Masked Autoencoders.]

## Acknowledgement

This repo is based on the official code of MAE. (https://github.com/facebookresearch/mae)

## The Uniformity-enhanced MAE loss

The main differences between our and the official code include:

1. We add a new uniformity regularizier, which is implemented in ``uni_reg.py``.
2. We add three hyperparemeters in ``pretrain.sh``. reg denotes the implementation form of the regularizer (default reg=spectral). $\beta$ and $\tau$ are designed to control the strength of the regularizier (default $tau=0.1$ and $beta = 0.0001$. 
3. We add a linear classfier to monitor the online linear accuracy whose gradient will not be backward propagated to the backbone.
