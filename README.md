# BCNN_DropCNN_classification
We employ the DiTing 2.0 dataset to develop two probabilistic deep learning models—Bayesian CNN (BCNN) and DropoutCNN—for discriminating between earthquakes and explosions. Both models provide uncertainty estimates for their classifications while maintaining comparable accuracy to traditional deterministic CNNs.

## Model Details:
src/cnn.py: A deterministic CNN for seismic discrimination.

src/model_bayesian_cnn.py: A probabilistic Bayesian CNN.

src/Dropcnn.py: A deterministic CNN enhanced with dropout layers for uncertainty estimation.

## Input Specifications:

All networks take waveform data of size 1×4000, where 1 represents the number of channels (single-channel input), and 4000 denotes the number of time-domain sampling points.

## Trained Models:

The following pretrained models are available:

model/BAYESIAN_CNN.pth: Trained Bayesian CNN.

model/CNN.pth: Baseline deterministic CNN.

model/Dropout CNNs with varying dropout rates:

DropCNNp0.05.pth (p=0.05)

DropCNNp0.1.pth (p=0.1)

DropCNNp0.2.pth (p=0.2)

DropCNNp0.3.pth (p=0.3)

DropCNNp0.4.pth (p=0.4)

DropCNNp0.5.pth (p=0.5)

These models were trained using model_bayesian_cnn, cnn.py, and Dropcnn.py with the respective dropout probabilities.

## Install conda and requirements
#### Download repository
```bash
git clone https://github.com/ZHANG-EP/BCNN_DropCNN_classification.git
cd BCNN_DropCNN_classification
```
#### Install to default environment
```bash
conda env update -f=env.yaml -n base
```
#### Install to "BCNN_DropCNN_classification" virtual envirionment
```bash
conda env create -f env.yaml
conda activate BCNN_DropCNN_classification
```
## Batch Prediction
```python
# please refer to the defaults.py file and the example_run.py file for detailed hyperparameter settings.
python ./src/example_run.py
```
## Related papers
Zhang, Yun, Li, Xihai, Zeng, Xiaoniu, Liu, Jihao, and Bai, Chaoying. Deep Learning Methods with Additional Confidence Metrics for Robust Earthquake vs. Explosion Classification. submitted to GJI (2025).

Any questions, please contact: zhangyun@chd.edu.cn.
