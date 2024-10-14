# CSE256 Fall 2024 Assignment 1 by Haozhou Xu
This project implements various sentiment classification models, including:

Bag of Words (BOW) models with two and three layers.
Deep Averaging Network (DAN) model using GloVe word embeddings.
A randomly initialized DAN model.
Byte Pair Encoding (BPE) based sentiment analysis model.
The goal is to train and evaluate these models on sentiment classification tasks using provided datasets, and to compare their performance in terms of accuracy.

## Authors
Haozhou Xu

## Requirements
Python 3.x
PyTorch
scikit-learn
matplotlib
re
To install the necessary dependencies, run:

```bash
pip install scikit-learn matplotlib torch re
```

Please also make sure to put the following files in the /data directory:
if no such directory exists, create one and put the following files in it.
```bash
mkdir data
mv dev.txt data/
mv train.txt data/
mv glove.6B.50d-relativized.txt data/
mv glove.6B.300d-relativized.txt data/
```

## Usage
You can run different models (BOW, DAN, random DAN, or BPE-DAN) by specifying the model type and options via command-line arguments.

Command-line Arguments:
* --model: Specify the model to run. Options include BOW, DAN, randDAN, or BPE.
* --freeze: Freeze the embeddings layer for the DAN model (default: Ture).
* --dropout: Set the dropout rate for models that use it (default: 0.3).
* --vocab: Set the vocabulary size for the BPE model (default: 500).

## Running the Models:
* BOW Model (with 2-layer and 3-layer neural networks):
```bash
python main.py --model BOW
```

* DAN Model (with GloVe embeddings):
```bash
python main.py --model DAN
# or
python main.py --model DAN --freeze False
```

* Random DAN Model (with randomly initialized embeddings):
```bash
python main.py --model RANDDAN
# or
python main.py --model RANDDAN --dropout 0.4
```

* BPE-DAN Model (using Byte Pair Encoding):
```bash
python main.py --model SUBWORDDAN
# or
python main.py --model SUBWORDDAN --vocab 1000
```

## File Structure
main.py: Contains the main function to run experiments and evaluate models.
sentiment_data.py: Functions to load embeddings.
utils.py: Utility functions for token idexing.
BOWmodels.py: Defines models based on Bag-of-Words representation.
DANmodels.py: Defines the Deep Averaging Network (DAN) models, including the BPE-based variant.
data/: Directory containing training and development datasets.
glove.6B.300d-relativized.txt: GloVe embeddings file used by DAN models.
Output
During training, the models will output the following:

Training and development accuracy printed after every epoch.
Training and development accuracy plots saved as PNG files:
train_accuracy_{ModelName}.png: Training accuracy for the selected model.
dev_accuracy_{ModelName}.png: Development accuracy for the selected model.
Example Output
After running the BOW model, you should see the training progress:

```yaml
Data loaded in : 2.134 seconds

2 layers:
Epoch #10: train accuracy 0.752, dev accuracy 0.725
Epoch #20: train accuracy 0.789, dev accuracy 0.743
...
```
The accuracy plots will be saved in the same directory as the script.