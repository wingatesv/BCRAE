# Breast Cancer Residual Autoencoder (BCRAE)

This repository contains python keras implementation of residual and transposed residual blocks for *Autoencoders* in breast cancer histopathological image task.

This repository contains:
1.  *RAE18*
2.  *RAE50*
3.  *RAE50v2*
4.  *Resnet18*
5.  *Resnet50*

# Acknowledgement
I would like to thank Kashiwa([@liao2000](https://liao2000.github.io/)) for the residual block guidance and tutorial.


# Reference:
https://machinelearningknowledge.ai/keras-implementation-of-resnet-50-architecture-from-scratch/

https://towardsdev.com/implement-resnet-with-tensorflow2-4ee81e33a1ac

https://github.com/liao2000/ML-Notebook/blob/main/ResNet/ResNet_TensorFlow.ipynb

If you find this repository helpful, please cite:

Voon, W.(2022), BCRAE. https://github.com/wingatesv/BCRAE.git (accessed date).


# Steps to use this repository

1.  Clone the repository to your workspace


```
! git clone https://github.com/wingatesv/BCRAE.git
```

2.  Import desired classes

```
import BCRAE.BC_RAE
from BCRAE.BC_RAE import RAE18, RAE50
```

3. Instantiate the model

```
rae18 = RAE18()
rae18 = rae18.model(input_shape=(224, 224, 3), name='RAE-18')
rae18.summary()
```
