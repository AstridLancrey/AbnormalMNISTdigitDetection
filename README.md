# AbnormalMNISTdigitDetection
# Exercise

We want to perform unsupervised anomaly detection on MNIST digit.
During training some digit classes (i.e. the anomalous one) are held out.
The model computes a score of normality (the higher the more normal).
At test time the model computes a score for all digit categories. 
The performance of the model is evaluated with the ROC AUC.

A Variational AutoEncoder (VAE) model loading and evaluation is provided for each digit.
10 models (one per abnormal digit) have already been trained for 30 epochs. All 10 models and 
their respective weights are saved as .pth files in the model_files Folder.

dataset.py file should be run in the 1st place:
If you have already downloaded the Mnistanomaly dataset in your current working repertory then you can
switch to False the download argument in the dataset.py file.

The main.py file consists in the initial code with a more performant model implemented: 
The execution of the this file enables to load for each digit the corresponding trained model
and to display the corresponding auc score in %. Running this main file will also generate 
3 output files per abnormal digit in the Results Folder (which is initially heamty before running main.py file)
Running main.py file will display the auc scores obtained with the 10 VAE models as well as the average auc
the same way that it initially did with the dumb model. 

The train.py file corresponds to the code that enabled to generate the 10 .pth model files. 

# Setup
## venv
To install the python virtualenv 
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements
```
## demo
To test the demonstration model:
```
python dataset.py
python main.py
```
It should print:
``` 
roc_auc per digit:
['0.919 ', '0.632 ', '0.905 ', '0.751 ', '0.667 ', '0.826 ', '0.907 ', '0.675 ', '0.812 ', '0.526 ']
average roc_auc:
0.762
```
It should save all output files in the Results Folder (3 by abnormal digit)
