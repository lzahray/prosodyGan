import numpy as np
import torch
import torch.nn as nn
import math
import os
import scipy.signal
from torch.utils.data import Dataset, random_split, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import midiutil

#https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b

#for masking this is maybe useful?
#but start with one specific length, let's do... well let's start with just the longest length in the batch. Pad with zeros.
#https://www.codeproject.com/Articles/5061271/PixelCNN-in-Autoregressive-Models

#https://arxiv.org/pdf/1611.09904.pdf
#this is terrible

class Generator(nn.Module):
    def __init__(self, input_channels, batch_size, device, hidden_size=200, num_layers=2, output_dim = 2, num_directions = 1):
        super(Generator, self).__init__()
        #was 25, 2
        #output dim 2 for pitch and onset, we concat emotion later
        #we're going to try a GRU
        self.device = device
        self.input_channels = input_channels
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.hidden = self.init_hidden()
        self.gru = nn.GRU(input_channels, hidden_size, num_layers, bidirectional=(self.num_directions==2), batch_first=True, dropout = 0.5)
        self.fc = nn.Linear(self.num_directions*self.hidden_size, output_dim) #was x2 before?
        self.activation = nn.Tanh()

    def init_hidden(self):
        return (torch.zeros(self.num_layers*self.num_directions, self.batch_size, self.hidden_size).to(self.device)) #x2 if bidirectional

    def forward(self, x, xlens):
        self.hidden = self.init_hidden()
        x = pack_padded_sequence(x.to(self.device), xlens, batch_first=True, enforce_sorted=False)
        output, self.hidden = self.gru(x, self.hidden)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.fc(output)
        return self.activation(output)

class Generator2(nn.Module):
    def __init__(self, input_channels, batch_size, device, noise_location = (0,4), emotion_location = (4,8), hidden_size=200, num_layers=2, output_dim = 2, num_directions = 1):
        super(Generator2, self).__init__()
        #was 25, 2
        #output dim 2 for pitch and onset, we concat emotion later
        #we're going to try a GRU
        self.device = device
        self.input_channels = input_channels
        self.input_size = self.input_channels + output_dim
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.output_dim = output_dim

        self.hidden = self.init_hidden()
        self.gru = nn.GRU(self.input_size, hidden_size, num_layers, bidirectional=(self.num_directions==2), batch_first=True, dropout = 0.5)
        self.fc = nn.Linear(self.num_directions*self.hidden_size, output_dim) #was x2 before?
        self.activation = nn.Tanh()

        self.noise_location = noise_location
        self.emotion_location = emotion_location

    def init_hidden(self):
        return (torch.zeros(self.num_layers*self.num_directions, self.batch_size, self.hidden_size).to(self.device)) #x2 if bidirectional

    def forward(self, x, xlens, targets = None):
        #go through each timestep individually.
        #Give data a start info vector - all 0s except noise and emotion
        #input: previous output, noise, emotions
        #print("x[0]: ", x[0,:,:])
        output = torch.zeros(x.size(0), x.size(1), self.output_dim).to(self.device)
        output[:,0,:] = 0
        self.hidden = self.init_hidden()
        initial_noise_and_emotion = torch.randn(x.size(0), 1, x.size(2)).to(self.device)
        initial_noise_and_emotion[:, :, -4:] = x[:, 0:1, -4:] #set emotions
        next_input = torch.cat((torch.ones((x.size(0), 1, self.output_dim)).to(self.device), initial_noise_and_emotion), axis=2).to(self.device)
        #print("first input[0]: ", next_input[0,:,:])
        for i in range(0, x.size(1)):
            output_temp, self.hidden = self.gru(next_input, self.hidden)
            fc_output = self.activation(self.fc(output_temp))
            output[:, i:i + 1, :] = fc_output
            #print("fc_output[0] ", fc_output[0,:,:])
            if targets is not None: #pretraining mode
                #print("pretrain")
                next_input = torch.cat((targets[:, i:i+1, :], x[:,i:i+1,:]), axis=2).to(self.device)
            else:
                #print("train")
                next_input = torch.cat((fc_output, x[:,i:i+1,:]), axis=2).to(self.device)
            #print("next_input[0] ", next_input[0,:,:])

        return output

class Baseline_Model(nn.Module):
    def __init__(self, input_channels, batch_size, device, noise_location = (0,4), emotion_location = (4,8), hidden_size=200, num_layers=2, output_dim = 2, num_directions = 1):
        super(Baseline_Model, self).__init__()
        #was 25, 2
        #output dim 2 for pitch and onset, we concat emotion later
        #we're going to try a GRU
        self.device = device
        self.input_channels = input_channels
        self.input_size = self.input_channels + output_dim
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.output_dim = output_dim

        self.hidden = self.init_hidden()
        self.gru = nn.GRU(self.input_size, hidden_size, num_layers, bidirectional=(self.num_directions==2), batch_first=True, dropout = 0.5)
        self.fc = nn.Linear(self.num_directions*self.hidden_size, output_dim) #was x2 before?
        self.activation = nn.Tanh()
        self.emotion_location = emotion_location

    def init_hidden(self):
        return (torch.zeros(self.num_layers*self.num_directions, self.batch_size, self.hidden_size).to(self.device)) #x2 if bidirectional

    def forward(self, x, xlens):
        #go through each timestep individually.
        #Give data a start info vector - all 0s except noise and emotion
        #input: previous output, noise, emotions
        #print("x[0]: ", x[0,:,:])
        output = torch.zeros(x.size(0), x.size(1), self.output_dim).to(self.device)
        targets = x[:,:,0:4]
        self.hidden = self.init_hidden()
        next_input = torch.cat((torch.ones((x.size(0), 1, self.output_dim)).to(self.device), x[:, 0:1, -4:]), axis=2).to(self.device)
        #print("first input[0]: ", next_input[0,:,:])
        for i in range(0, x.size(1)):
            output_temp, self.hidden = self.gru(next_input, self.hidden)
            fc_output = self.activation(self.fc(output_temp))
            output[:, i:i + 1, :] = fc_output
            #print("fc_output[0] ", fc_output[0,:,:])
                #print("pretrain")
            next_input = torch.cat((targets[:, i:i+1, :], x[:, 0:1, -4:]), axis=2).to(self.device)

        return output

class Discriminator(nn.Module):
    def __init__(self, input_channels, batch_size, device, hidden_size=200, num_layers=2, num_directions = 2):
        super(Discriminator, self).__init__()
        #was 35, 2
        self.device = device
        self.input_channels = input_channels
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.hidden = self.init_hidden()
        self.gru = nn.GRU(input_channels, hidden_size, num_layers, bidirectional=(self.num_directions==2), batch_first=True)
        self.dense = nn.Linear(self.hidden_size*self.num_directions, 1)
        self.activation = nn.Tanh()

    def init_hidden(self):
        return (torch.zeros(self.num_layers*self.num_directions, self.batch_size, self.hidden_size).to(self.device))

    def forward(self, x):
        self.hidden = self.init_hidden()
        output, self.hidden = self.gru(x.to(self.device), self.hidden)
        final_state = self.hidden.view(self.num_layers, self.num_directions, self.batch_size, self.hidden_size)[-1]
        if self.num_directions == 2:
            h_1, h_2 = final_state[0], final_state[1]
            X = torch.cat((h_1,h_2), 1)
        else:
            X = final_state.squeeze()
        output_last = self.dense(X)
        return output_last
        #return self.activation(output_last)


class Classifier(nn.Module):
    def __init__(self, input_channels, batch_size, device, output_dim, hidden_size=200, num_layers=2, num_directions = 2):
        super(Classifier, self).__init__()
        #was 35, 2
        self.device = device
        self.input_channels = input_channels
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.hidden = self.init_hidden()
        self.gru = nn.GRU(input_channels, hidden_size, num_layers, bidirectional=(self.num_directions==2), batch_first=True)
        self.output_dim = output_dim
        self.dense = nn.Linear(self.hidden_size*self.num_directions, self.output_dim) #embed the 4-dim emotion thing
        #self.activation = nn.Tanh()
        #self.activation = nn.Softmax()


    def init_hidden(self):
        return (torch.zeros(self.num_layers*self.num_directions, self.batch_size, self.hidden_size).to(self.device))

    def forward(self, x):
        self.hidden = self.init_hidden()
        output, self.hidden = self.gru(x.to(self.device), self.hidden)
        final_state = self.hidden.view(self.num_layers, self.num_directions, self.batch_size, self.hidden_size)[-1]
        if self.num_directions == 2:
            h_1, h_2 = final_state[0], final_state[1]
            X = torch.cat((h_1,h_2), 1)
        else:
            X = final_state.squeeze()
        output_last = self.dense(X)
        return output_last
        #return self.activation(output_last)



