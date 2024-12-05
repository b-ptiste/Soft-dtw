# Work overview

The report delves into Soft-Dynamic Time Warping (Soft-DTW), a differentiable version of Dynamic Time Warping, suitable for gradient-based optimization in machine learning. It involves reimplementation of the algorithm, theoretical and practical analysis, and experimentation with datasets like ArrowHead and ECG200. The findings include :

- PyTorch-compatible loss with custom backward for efficiency purposes 🚀 : **Soft-DTW**
- Applications :
  - Barycenter averaging
  - K-Means clustering
  - Anomaly detection

While the report concludes with the greate applications of the Soft-DTW loss, the time complexity is too large. It suggests directions for future research.

# Simple utilisation 

Our code is compatible with any native **Pytorch** implementation. 

We **re-implemented the backward** for efficiency purposes 🚀.

```python
import torch
from tslearn.datasets import UCR_UEA_datasets
from DTWLoss_CUDA import DTWLoss

# load data
ucr = UCR_UEA_datasets()
X_train, y_train, X_test, y_test = ucr.load_dataset("SonyAIBORobotSurface2")

# convert to torch
X_train = torch.from_numpy(X_train).float().requires_grad_(True)
loss = DTWLoss(gamma=0.1)
optimizer = # your optimizer

##############
# your code ##
##############

value = loss(X_train[0].unsqueeze(0), X_train[1].unsqueeze(0))
optimizer.zero_grad()
value.backward()
optimizer.step()
```

# Nice Experiments

## Avering times series
![avering](https://github.com/b-ptiste/dtw-soft/assets/75781257/b1373a3a-f1b7-4ea3-8701-912d511f7c72)

## K-MEANS
![Capture d'écran 2024-01-09 114025](https://github.com/b-ptiste/dtw-soft/assets/75781257/02cdacde-e02b-42f1-afaa-8954730e1fe9)

## Anomaly detection
![Capture d'écran 2024-01-09 114258](https://github.com/b-ptiste/dtw-soft/assets/75781257/e1c1702a-8952-4fc7-a2e1-af74c60e94de)

# Authors : 
- CALLARD Baptiste (MVA)
- DE SENNEVILLE Adhemar (MVA)

# Credit

Soft-dtw: a differentiable loss function for time-series by Cuturi, Marco and Blondel, Mathieu in International conference on machine learning
