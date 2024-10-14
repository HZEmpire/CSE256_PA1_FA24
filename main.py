# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples, WordEmbeddings, read_word_embeddings
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from DANmodels import SentimentDatasetDAN, DAN, BPE


# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')
    parser.add_argument('--freeze', type=bool, default=True, help='Freeze word embeddings for DAN model')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate for the model')
    parser.add_argument('--vocab', type=int, default=500, help='Vocabulary size for BPE model')
    # Parse the command-line arguments
    args = parser.parse_args()

    # Check if the model type is "BOW"
    if args.model == "BOW":
        # Load dataset
        start_time = time.time()

        train_data = SentimentDatasetBOW("data/train.txt")
        dev_data = SentimentDatasetBOW("data/dev.txt")
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")

        # Train and evaluate NN2
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Training and evaluation completed in : {elapsed_time} seconds")

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

    elif args.model == "DAN":
        # Load dataset
        start_time = time.time()

        word_embeddings = read_word_embeddings("data/glove.6B.300d-relativized.txt")
        train_data = SentimentDatasetDAN("data/train.txt", word_embeddings)
        dev_data = SentimentDatasetDAN("data/dev.txt", word_embeddings)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)
        

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")

        # Train and evaluate DAN
        start_time = time.time()
        dan_train_accuracy, dan_test_accuracy = experiment(DAN(wordEmbeddings=word_embeddings, freeze=args.freeze), train_loader, test_loader)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Training and evaluation completed in : {elapsed_time} seconds")

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_train_accuracy, label='DAN')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for DAN')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy_dan.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_test_accuracy, label='DAN')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for DAN')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy_dan.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

    
    elif args.model == "RANDDAN":
        # Load dataset
        start_time = time.time()

        word_embeddings = read_word_embeddings("data/glove.6B.50d-relativized.txt")
        train_data = SentimentDatasetDAN("data/train.txt", word_embeddings)
        dev_data = SentimentDatasetDAN("data/dev.txt", word_embeddings)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)
        

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")

        # Train and evaluate randDAN
        start_time = time.time()
        vocab_size = len(word_embeddings.word_indexer)
        embedding_dim = word_embeddings.get_embedding_length()
        randdan_train_accuracy, randdan_test_accuracy = experiment(DAN(vocab_size=vocab_size, embedding_dim=embedding_dim, dropout=args.dropout), train_loader, test_loader)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Training and evaluation completed in : {elapsed_time} seconds")

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(randdan_train_accuracy, label='Random DAN')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for Random DAN')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy_randdan.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(randdan_test_accuracy, label='Random DAN')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for Random DAN')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy_randdan.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

    elif args.model == "SUBWORDDAN":
        # Load dataset
        start_time = time.time()

        sentences = read_sentiment_examples("data/train.txt")
        bpe = BPE(vocab_size=args.vocab)
        bpe.fit([" ".join(ex.words) for ex in sentences])
        train_data = SentimentDatasetDAN("data/train.txt", word_embeddings=None, bpe=bpe)
        dev_data = SentimentDatasetDAN("data/dev.txt", word_embeddings=None, bpe=bpe)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)
        

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")

        # Train and evaluate randDAN
        start_time = time.time()
        vocab_size = len(bpe.idx_to_obj)
        embedding_dim = 50
        randdan_train_accuracy, randdan_test_accuracy = experiment(DAN(vocab_size=vocab_size, embedding_dim=embedding_dim, dropout=0.3), train_loader, test_loader)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Training and evaluation completed in : {elapsed_time} seconds")

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(randdan_train_accuracy, label='BPE DAN')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for BPE DAN')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy_bpedan.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(randdan_test_accuracy, label='BPE DAN')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for BPE DAN')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy_bpedan.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")



    else:
        print(f"Unknown model type: {args.model}")
    

if __name__ == "__main__":
    main()
