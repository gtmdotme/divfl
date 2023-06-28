# Diverse Client Selection for Federated Learning via Submodular Maximization

## Getting Started Guide
* **Python 3.10**
* **CUDA: 11.8**
* **CuDNN: 8.6.0**
* **TensorFlow 2.12.0**

Step 1: Log into Scholar Cluster and load modules
* `$ ssh username@gpu.scholar.rcac.purdue.edu`
* `$ module purge`
* `$ module load cuda/11.8.0`
* `$ module load cudnn`
* `$ module load anaconda/2020.11-py38`

Step 2: Create or activate conda environment
* `$ conda create -n myenv python=3.10 ipython ipykernel -y`
* `$ conda activate myenv`
* `$ python -m ipykernel install --user --name myenv --display-name "Python (myenv)"`

Step 3.2: Install **TensorFlow 2.12.0**
* May refer: https://www.tensorflow.org/install/pip#linux
* For Ubuntu:
```bash
$ conda install -c conda-forge cudatoolkit=11.8.0 -y
$ python -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
$ mkdir -p $CONDA_PREFIX/etc/conda/activate.d
$ echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
$ echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
$ source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```
* For Mac: `$ pip install tensorflow` (may install 2.13.0rc1)
* Verify Install:
```bash
$ python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}\nCUDA devices: {tf.config.list_physical_devices(\"GPU\")}')"
```

Step 3.4: Install common packages
* $ ML Toolkits: `$ conda install -c anaconda pandas numpy scikit-learn scipy h5py networkx xgboost seaborn tqdm tabulate pillow -y`
* $ Misc: `conda install -c conda-forge loguru jupyterlab -y`
* Verify install:
```bash
$ python -c "import pandas, numpy, requests, networkx, seaborn, tqdm, matplotlib, sklearn, loguru; print ('Done')"
```

## Data Directory
You can replace the `data` directory by downloading from [here](https://purdue0-my.sharepoint.com/:u:/g/personal/gchoudha_purdue_edu/ESL4_QIds8JLk36JqVXKNKkBGey4fWQGPaqESJjRYujs-w?e=JmJcnS) which contains all the processed data (except for CelebA).


-----
# Old Readme


## Code for ICLR 2022 paper:

<b>Title</b>: <i>Diverse Client Selection for Federated Learning via Submodular Maximization</i> <a href="https://openreview.net/pdf?id=nwKXyFvaUm">[pdf]</a> <a href="https://iclr.cc/virtual/2022/poster/7047">[presentation]</a>\
<b>Authors</b>: Ravikumar Balakrishnan* (Intel Labs), Tian Li* (CMU), Tianyi Zhou* (UW), Nageen Himayat (Intel Labs), Virginia Smith (CMU), Jeff Bilmes (UW)\
<b>Institutes</b>: Intel Labs, Carnegie Mellon University, University of Washington

<pre>
@inproceedings{
balakrishnan2022diverse,
title={Diverse Client Selection for Federated Learning via Submodular Maximization},
author={Ravikumar Balakrishnan and Tian Li and Tianyi Zhou and Nageen Himayat and Virginia Smith and Jeff Bilmes},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=nwKXyFvaUm}
}</pre>


<b>Abstract</b>\
In every communication round of federated learning, a random subset of clients communicate their model updates back to the server which then aggregates them all. The optimal size of this subset is not known and several studies have shown that typically random selection does not perform very well in terms of convergence, learning efficiency and fairness. We, in this paper, propose to select a small diverse subset of clients, namely those carrying representative gradient information, and we transmit only these updates to the server. Our aim is for updating via only a subset to approximate updating via aggregating all client information. We achieve this by choosing a subset that maximizes a submodular facility location function defined over gradient space. We introduce “federated averaging with diverse client selection (DivFL)”. We provide a thorough analysis of its convergence in the heterogeneous setting and apply it both to synthetic and to real datasets. Empirical results show several benefits to our approach including improved learning efficiency, faster convergence and also more uniform (i.e., fair) performance across clients. We further show a communication-efficient version of DivFL that can still outperform baselines on the above metrics.

## Preparation

### Dataset generation

We **already provide four synthetic datasets** that are used in the paper under corresponding folders. For all datasets, see the `README` files in separate `data/$dataset` folders for instructions on preprocessing and/or sampling data.

The statistics of real federated datasets are summarized as follows.

<center>

| Dataset       | Devices         | Samples|Samples/device <br> mean (stdev) |
| ------------- |-------------| -----| ---|
| MNIST      | 1,000 | 69,035 | 69 (106)| 
| FEMNIST     | 200      |   18,345 | 92 (159)|
| Shakespeare | 143    |    517,106 | 3,616 (6,808)|
| Sent140| 772      |    40,783 | 53 (32)|

</center>

### Downloading dependencies

```
pip3 install -r requirements.txt  
```

## References
See our [DivFL](https://openreview.net/pdf?id=nwKXyFvaUm) paper for more details as well as all references.

## Acknowledgements
Our implementation is based on [FedProx](https://github.com/litian96/FedProx).
