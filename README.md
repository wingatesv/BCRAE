# Breast Cancer Residual Autoencoder (BCRAE)

This repository contains python residual and transposed residual blocks to build Autoencoders (RAE-18 and RAE-50) for breast cancer histopathological image task.


Reference:
https://machinelearningknowledge.ai/keras-implementation-of-resnet-50-architecture-from-scratch/
https://github.com/liao2000/ML-Notebook/blob/main/ResNet/ResNet_TensorFlow.ipynb


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
