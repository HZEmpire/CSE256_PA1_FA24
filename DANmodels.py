# Author: Haozhou Xu
# PID: A69032157

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import WordEmbeddings
from utils import Indexer
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset

# Dataset class for handling sentiment analysis data
import torch
from torch.utils.data import Dataset
from sentiment_data import read_sentiment_examples
from utils import Indexer

class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, word_embeddings: WordEmbeddings=None):
        """
        The constructor for SentimentDatasetDAN
        params:
            infile: the path to the input file
            word_embeddings: the word embeddings object
        """
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile)
        
        # Extract sentences and labels from the examples
        self.sentences = [" ".join(ex.words) for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]

        # Convert sentences to word indices using the Indexer
        self.sentences_idx = []
        for sentence in self.sentences:
            word_indices = [word_embeddings.word_indexer.index_of(word) for word in sentence.split()]
            # Replace all -1 indices with 1
            word_indices = [1 if idx == -1 else idx for idx in word_indices]
            self.sentences_idx.append(word_indices)

        # Pad the sentences to the maximum length
        max_length = max([len(sentence) for sentence in self.sentences_idx])
        for i in range(len(self.sentences_idx)):
            self.sentences_idx[i] += [0] * (max_length - len(self.sentences_idx[i]))
        
        # Convert embeddings and labels to PyTorch tensors
        self.sentences_idx = torch.tensor(self.sentences_idx, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
    
    def __len__(self):
        """The length of the dataset
        returns:
            the number of examples in the dataset
        """
        return len(self.examples)

    def __getitem__(self, idx):
        """Get an example from the dataset
        params:
            idx: the index of the example
        returns:
            the word indices of the example and the label
        """
        return self.sentences_idx[idx], self.labels[idx]

# Deep Averaging Network
class DAN(nn.Module):
    def __init__(self, wordEmbeddings=None, hidden_size=200, freeze=True, dropout=0.6, vocab_size=None, embedding_dim=None):
        """The constructor for DAN
        params:
            wordEmbeddings: the word embeddings object
            hidden_size: the size of the hidden layer
            freeze: whether to freeze the embeddings
        """
        super().__init__()
        if wordEmbeddings is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.fc1 = nn.Linear(embedding_dim, hidden_size)
        else:
            self.embedding = wordEmbeddings.get_initialized_embedding_layer(freeze)
            self.fc1 = nn.Linear(wordEmbeddings.get_embedding_length(), hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """The forward pass of DAN
        params:
            x: the input indices tensor
        returns:
            the output prediction tensor
        """
        x = self.embedding(x.long())
        x = torch.mean(x, dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return self.log_softmax(x)
