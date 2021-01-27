import os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
import tqdm
import pickle

# preprocessing
def prepare_data(data_path):
    with open(data_path) as f:
        # This reads all the data from the file, but does not do any processing on it.
        data = f.read()
    
    # Add more preprocessing
    data.replace("\n", " ")
    data.replace("\t", " ")
    #data = data.lower()
    voc2ind = {}
    curridx = 0
    idxdata = []
    
    # Compute voc2ind and transform the data into an integer representation of the tokens.
    for char in data:
        if char not in voc2ind:
          voc2ind[char] = curridx
          curridx +=1
        idxdata.append(voc2ind[char])


    ind2voc = {val: key for key, val in voc2ind.items()}
    splitidx = int(len(idxdata)*0.8)

    train_text = idxdata[:splitidx]
    test_text = idxdata[splitidx:]

    pickle.dump({'tokens': train_text, 'ind2voc': ind2voc, 'voc2ind':voc2ind}, open(DATA_PATH + 'harry_potter_chars_train.pkl', 'wb'))
    pickle.dump({'tokens': test_text, 'ind2voc': ind2voc, 'voc2ind':voc2ind}, open(DATA_PATH + 'harry_potter_chars_test.pkl', 'wb'))
    
#prepare_data(DATA_PATH + '<<language_name>>.txt')


class Vocabulary(object):
    def __init__(self, data_file):
        with open(data_file, 'rb') as data_file:
            dataset = pickle.load(data_file)
        self.ind2voc = dataset['ind2voc']
        self.voc2ind = dataset['voc2ind']

    # Returns a string representation of the tokens.
    def array_to_words(self, arr):
        return ''.join([self.ind2voc[int(ind)] for ind in arr])

    # Returns a torch tensor representing each token in words.
    def words_to_array(self, words):
        return torch.LongTensor([self.voc2ind[word] for word in words])

    # Returns the size of the vocabulary.
    def __len__(self):
        return len(self.voc2ind)




class LanguageDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, sequence_length, batch_size):
        super(LanguageDataset, self).__init__()

        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.vocab = Vocabulary(data_file)

        with open(data_file, 'rb') as data_pkl:
            dataset = pickle.load(data_pkl)
        self.data = dataset['tokens'][:int(len(dataset['tokens'])/batch_size)*batch_size]


    def __len__(self):
        return int( int(len(self.data)/self.batch_size-2) /self.sequence_length+1)*self.batch_size
        
    def __getitem__(self, idx):
        # Return the data and label for a character sequence as described above.
        # The data and labels should be torch long tensors.
        # You should return a single entry for the batch using the idx to decide which chunk you are 
        # in and how far down in the chunk you are.
        chunk_num = idx % self.batch_size
        chunk_ind = idx // self.batch_size * self.sequence_length
        chunk_size = int(len(self.data) / self.batch_size)
        chunk_end = (chunk_num + 1) * chunk_size
        begin = int(chunk_num * chunk_size + chunk_ind)
        end = int(min(begin + self.sequence_length+1, chunk_end))
        data = torch.LongTensor(self.data[begin:end])
        return data[:-1], data[1:]

    def vocab_size(self):
        return len(self.vocab)


class CharNet(nn.Module):
    def __init__(self, vocab_size, feature_size):
        super(CharNet, self).__init__()
        self.vocab_size = vocab_size
        self.feature_size = feature_size
        self.encoder = nn.Embedding(self.vocab_size, self.feature_size)
        self.gru = nn.GRU(self.feature_size, self.feature_size, batch_first=True)
        self.decoder = nn.Linear(self.feature_size, self.vocab_size)
        
        # This shares the encoder and decoder weights
        self.decoder.weight = self.encoder.weight
        self.decoder.bias.data.zero_()
        
        self.best_accuracy = -1
    
    def forward(self, x, hidden_state=None):
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        
        # defining the forward pass.
        # return the output from the decoder as well as the hidden state given by the gru.
        x = self.encoder(x)
        x, hidden_state = self.gru(x, hidden_state)
        x = self.decoder(x)

        return x, hidden_state

    # This returns the top 3 characters
    def inference(self, x, hidden_state=None, temperature=1):
        x = x.view(-1, 1)
        x, hidden_state = self.forward(x, hidden_state)
        x = x.view(1, -1)
        x = x / max(temperature, 1e-20)
        x = F.softmax(x, dim=1)
        return x, hidden_state

    # Predefined loss function
    def loss(self, prediction, label, reduction='mean'):
        loss_val = F.cross_entropy(prediction.view(-1, self.vocab_size), label.view(-1), reduction=reduction)
        return loss_val

    # Saves the current model
    def save_model(self, file_path):
        raise NotImplementedError

    # Saves the best model so far
    def save_best_model(self, accuracy, file_path):
        if accuracy > self.best_accuracy:
            self.save_model(file_path)
            self.best_accuracy = accuracy

    # load checkpoint model
    def load_model(self, file_path):
        raise NotImplementedError