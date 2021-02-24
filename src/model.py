import os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
import torch.optim as optim
import tqdm
import pickle
import random

from language_cleaner import *

DATA_PATH = './data/'

# preprocessing
def load_training_data(filename, language):
    rd = RussianCleaner('./data/russian/interfax.csv', save_file_path="/local1/sharmava-nlp-data/russian.txt", save_sentences=True)
    sd = SpanishCleaner('./data/spanish/', save_file_path="/local1/sharmava-nlp-data/spanish.txt", save_sentences=True)
    
    with open('/local1/sharmava-nlp-data/portuguese.txt') as f:
        pd = f.read()
    
    with open('/local1/sharmava-nlp-data/english.txt') as f:
        ed = f.read()
    len_ed = len(ed)
    ed = ed[:int(len_ed * 0.05)]

    with open('/local1/sharmava-nlp-data/chinese.txt') as f:
        cd = f.read()
    
    arabic = WortschatzLanguageParser('data/ara_news_2017_1M-sentences.txt', '/local1/sharmava-nlp-data/arabic.txt', True)
    dutch = WortschatzLanguageParser('data/deu_mixed-typical_2011_1M-sentences.txt', '/local1/sharmava-nlp-data/dutch.txt', True)
    french = WortschatzLanguageParser('data/fra_newscrawl-public_2019_1M-sentences.txt', '/local1/sharmava-nlp-data/french.txt', True)
    luxemborgish = WortschatzLanguageParser('data/ltz-lu_web_2013_1M-sentences.txt', '/local1/sharmava-nlp-data/luxemborgish.txt', True)

    arabic_sd = arabic.get_sentence_list()
    dutch_sd = dutch.get_sentence_list()
    french_sd = french.get_sentence_list()
    luxemborgish_sd = luxemborgish.get_sentence_list()

    sd_min = sd.get_sentence_list()
    sd_min = sd_min[:int(len(sd_min) * 0.33)]

    rd_min = rd.get_sentence_list()
    rd_min = rd_min[:int(len(rd_min) * 0.2)]

    sentence_list = []
    sentence_list.extend(sd_min)
    sentence_list.extend(rd_min)
    sentence_list.extend(arabic_sd)
    sentence_list.extend(dutch_sd)
    sentence_list.extend(french_sd)
    sentence_list.extend(luxemborgish_sd)
    
    random.shuffle(sentence_list) # shuffle the 1d list so the languages are mixed up!

    data =""
    data += ' '.join(sentence_list)
    data += ' '.join(pd)
    data += ' '.join(ed)
    data += ' '.join(cd)
    
    # Add more preprocessing
    data.replace("\n", " ")
    data.replace("\t", " ")
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

    pickle.dump({'tokens': train_text, 'ind2voc': ind2voc, 'voc2ind':voc2ind}, open(DATA_PATH + language + '_chars_train.pkl', 'wb'))
    pickle.dump({'tokens': test_text, 'ind2voc': ind2voc, 'voc2ind':voc2ind}, open(DATA_PATH + language + '_chars_test.pkl', 'wb'))
    
    return


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
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    def __init__(self, vocab, feature_size):
        super(CharNet, self).__init__()
        self.vocab_size = len(vocab)
        self.feature_size = feature_size
        self.encoder = nn.Embedding(self.vocab_size, self.feature_size)
        self.gru = nn.GRU(self.feature_size, self.feature_size, batch_first=True)
        self.decoder = nn.Linear(self.feature_size, self.vocab_size)
        
        # This shares the encoder and decoder weights
        self.decoder.weight = self.encoder.weight
        self.decoder.bias.data.zero_()
        
        self.best_accuracy = -1
        self.vocab = vocab
    
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
        x = x.view(1, -1)
        x, hidden_state = self.forward(x, hidden_state)
        x = x[0,-1].view(1, -1)
        res, idx = torch.topk(x, 3)


        return [self.vocab.ind2voc[int(ind)] for ind in idx.tolist()[0]]
    
    # Predefined loss function
    def loss(self, prediction, label, reduction='mean'):
        loss_val = F.cross_entropy(prediction.view(-1, self.vocab_size), label.view(-1), reduction=reduction)
        return loss_val

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname, encoding='utf-8') as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_pred(self, data):
        # your code here
        preds = []
        USE_CUDA = True
        use_cuda = USE_CUDA and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        for inp in data:
            # this model just predicts a random character each time
            data = [self.vocab.voc2ind[c] if c in self.vocab.voc2ind else random.randint(0, self.vocab_size-1) for c in inp]
            data = torch.LongTensor(data).to(device)
            top_guesses = self.inference(data)
            preds.append(''.join(top_guesses))
        return preds

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        #with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
        #    f.write(torch.save(model.state_dict(), PATH))
        torch.save(self, work_dir + 'model.checkpoint.pt')

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        #with open(os.path.join(work_dir, 'model.checkpoint')) as f:
        #    dummy_save = f.read()
        if torch.cuda.is_available():
            model = torch.load(work_dir + '/model.checkpoint.pt')
        else:
            model = torch.load(work_dir + '/model.checkpoint.pt', map_location = torch.device('cpu'))
        return model