# BCNN_DropCNN_classification
We employ the DiTing 2.0 dataset to develop two probabilistic deep learning models—Bayesian CNN (BCNN) and DropoutCNN—for discriminating between earthquakes and explosions. Both models provide uncertainty estimates for their classifications while maintaining comparable accuracy to traditional deterministic CNNs.

Model Details:
cnn.py: A deterministic CNN for seismic discrimination.

model_bayesian_cnn: A probabilistic Bayesian CNN.

Dropcnn.py: A deterministic CNN enhanced with dropout layers for uncertainty estimation.

Input Specifications:

All networks take waveform data of size 1×4000, where 1 represents the number of channels (single-channel input), and 4000 denotes the number of time-domain sampling points.

Trained Models:

The following pretrained models are available:

BAYESIAN_CNN.pth: Trained Bayesian CNN.

CNN.pth: Baseline deterministic CNN.

Dropout CNNs with varying dropout rates:

DropCNNp0.05.pth (p=0.05)

DropCNNp0.1.pth (p=0.1)

DropCNNp0.2.pth (p=0.2)

DropCNNp0.3.pth (p=0.3)

DropCNNp0.4.pth (p=0.4)

DropCNNp0.5.pth (p=0.5)

These models were trained using model_bayesian_cnn, cnn.py, and Dropcnn.py with the respective dropout probabilities.

Any questions, please contact: zhangyun@chd.edu.cn.
