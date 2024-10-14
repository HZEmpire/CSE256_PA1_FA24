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
import re
from collections import defaultdict


# Class for the BPE embeddings
class BPE:
    def __init__(self, vocab_size=1000):
        """The constructor for BPE
        params:
            vocab_size: the size of the vocabulary
        """
        self.vocab_size = vocab_size
        self.vocab = defaultdict(int)
        self.obj_to_idx = {'</w>': 2}
        self.idx_to_obj = {2: '</w>'}

    def get_stats(self, vocab):
        """Find the pairs of characters
        params:
            vocab: the vocabulary
        returns:
            the pairs of characters
        """
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_vocab(self, pair):
        """Merge the vocabulary
        params:
            pair: the pair of characters
        """
        v_out = {}
        bigram = re.escape (' '.join(pair)) 
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)') 
        for word in self.vocab: 
            w_out = p.sub (''. join(pair), word) 
            v_out[w_out] = self.vocab[word] 
        self.vocab = v_out

    def fit(self, sentences):
        """Fit the BPE model
        params:
            sentences: the list of sentences
        """
        # Initialize the vocabulary
        self.vocab = defaultdict(int)
        for sentence in sentences:
            words = sentence.strip().split()
            for word in words:
                for each in list(word):
                    if each not in self.obj_to_idx:
                        self.obj_to_idx[each] = len(self.obj_to_idx) + 2
                        self.idx_to_obj[len(self.idx_to_obj) + 2] = each
                word = ' '.join(list(word)) + ' </w>'
                self.vocab[word] += 1

        # Get the BPE codes
        while len(self.obj_to_idx) < self.vocab_size:
            pairs = self.get_stats(self.vocab)
            # Reached maximum vocab size
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.idx_to_obj[len(self.idx_to_obj) + 2] = ''.join(best)
            self.obj_to_idx[''.join(best)] = len(self.obj_to_idx) + 2
            self.merge_vocab(best)
            if len(self.obj_to_idx) % 500 == 0:
                print(f"Finished BPE iteration {len(self.obj_to_idx) / self.vocab_size * 100}%")
        self.obj_to_idx['UNK'] = 1
        self.idx_to_obj[1] = 'UNK'
        self.obj_to_idx['PAD'] = 0
        self.idx_to_obj[0] = 'PAD'

    def encode(self, word):
        """Encode the word
        params:
            word: the input word
        returns:
            the encoded word
        """
        word = ' '.join(list(word)) + ' </w>'
        while True:
            pairs = self.get_stats({word: 1})
            if not pairs:
                break
            flag = False
            while pairs:
                best = max(pairs, key=pairs.get)
                if ''.join(best) in self.obj_to_idx:
                    word = word.replace(' '.join(best), ''.join(best))
                    flag = True
                pairs.pop(best)
            if not flag:
                break
        return word.split()
    
    def tokenize(self, sentence):
        """Tokenize the sentence
        params:
            sentence: the input sentence
        returns:
            the tokenized sentence
        """
        words = sentence.strip().split()
        tokens = []
        for word in words:
            tokens.extend(self.encode(word))
        return tokens
    
    def index_of(self, token):
        """Get the index of the token
        params:
            token: the input token
        returns:
            the index of the token
        """
        if token in self.obj_to_idx:
            return self.obj_to_idx[token]
        return 1

    def obj_of(self, idx):
        """Get the object of the index
        params:
            idx: the input index
        returns:
            the object of the index
        """
        if idx in self.idx_to_obj:
            return self.idx_to_obj[idx]
        return 'UNK'
       
class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, word_embeddings: WordEmbeddings=None, bpe: BPE=None):
        """
        The constructor for SentimentDatasetDAN
        params:
            infile: the path to the input file
            word_embeddings: the word embeddings object
            bpe: the BPE object
        """
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile)
        
        # Extract sentences and labels from the examples
        self.sentences = [" ".join(ex.words) for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]

        # Convert sentences to word indices using the Indexer
        self.sentences_idx = []
        if bpe is None:
            for sentence in self.sentences:
                word_indices = [word_embeddings.word_indexer.index_of(word) for word in sentence.split()]
                # Replace all -1 indices with 1
                word_indices = [1 if idx == -1 else idx for idx in word_indices]
                self.sentences_idx.append(word_indices)
        else:
            for sentence in self.sentences:
                tokens = bpe.tokenize(sentence)
                bpe_indices = [bpe.index_of(token) for token in tokens]
                self.sentences_idx.append(bpe_indices)

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

def main():
    sentences = read_sentiment_examples("data/train.txt")
    bpe = BPE(vocab_size=500)
    bpe.fit([" ".join(ex.words) for ex in sentences])
    tokens = bpe.tokenize("your test sentence here")
    #print(bpe.idx_to_obj)
    #print(bpe.obj_to_idx)
    #print(bpe.vocab)
    print(tokens)

if __name__ == "__main__":
    main()