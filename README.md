# Auto-CO-AFM
An automated solution for carbon monoxide functionalization which combinesmachine learning descriptors with automated software control of the tip preparation process. 

![Schematic](/images/CO-tip-evaluator.png)


 The machine learning models are implemented in Tensorflow 1.12. The code is currently written in Python 3. At least the following Python packages are required:
* numpy
* matplotlib
* tensorflow-gpu=1.12.0
* jupyter

Additionally, you need to have Cuda and cuDNN correctly configured on your system in order to train the models on an Nvidia GPU.
## Database
AFM-data with CO-tips samples can be downloaded [here](https://www.dropbox.com/s/wqhgbvdlfb6snic/datasetNew6.tar.gz?dl=0). 

## Installation

If you are using Anaconda, you can create the required Python environment with
```sh
conda env create -f environment.yml
```
This will create a conda enviroment named tf-gpu with the all the required packages. It also has a suitable version of the Cuda toolkit and cuDNN already installed. Activate the environment with
```sh
conda activate py3-tf12
```

To create the datasets and train the models, run `jupyter notebook` in the repository folder, open the `train_TF.ipynb` notebook, and follow the instructions therein.

The folder `pretrained_weights` holds the weights for pretrained model.

To predict quality of CO-tip on some set of images of CO tips, open the `predict_TF.ipynb` notebook, and follow the instructions therein. Good CO tips predicted as 1, bads COs as 0.

To perform autonomous CO functionalization, open the `auto-co.ipynb` notebook, and follow the instructions therein. Ensure that your CreaTec STM is already connected and that COM support is enabled during CreaTec STMAFM software installation.
