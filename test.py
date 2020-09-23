import numpy as np
import torch
import torch.nn as nn
import math
import os
from torch.utils.data import Dataset, random_split, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from model import Generator, Discriminator, Generator2
from utils import EmotionDataset, MidiWriter, EmotionDatasetTiming



device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print("device: ", device)
generator_input_channels = 4 + 4
batch_size = 4
test_model = Generator2(generator_input_channels, batch_size, device,output_dim=4).to(device)
checkpoint = torch.load("checkpoints_gpTime/generator_checkpoint184.pth.tar") #54 wasn't bad I think
test_model.load_state_dict(checkpoint['model_state_dict'])
test_model.eval()
test_model.to(device)
emotion = "admiration"
time = 6*20

noise = torch.randn(batch_size, time, 4).to(device)
dataset = EmotionDatasetTiming("../prosodyProject/processed_midi_info_edited/", 20,device)
tiled_emotions = np.tile(dataset.emotion_embeddings[emotion], (noise.size()[1], 1))
tiled_emotions = np.tile(np.expand_dims(tiled_emotions, 0),(batch_size,1,1))
#print("tiled_emotions.shape ", tiled_emotions.shape)
#print("first block of emotions ", tiled_emotions[0,0:3, :])
tiled_emotions = torch.from_numpy(tiled_emotions).float().to(device)
initial_sequence = torch.cat((noise,tiled_emotions), axis=2).to(device)
#print("initial_sequence size is ", initial_sequence.size())
#print("initial_sequence was : ", initial_sequence)
generated_midi = test_model(initial_sequence, [initial_sequence.size()[1]])
print("pitches orig: ", generated_midi[0,:,0])
print("durations orig: ",generated_midi[0,:,1])
print("time since onset orig: ", generated_midi[0,:,2])
print("onsets orig: ",generated_midi[0,:,3])
# #dataset.create_midi_file(generated_midi[0,:,0].detach().cpu().numpy(), "newMidiRelief1")
midiWriter = MidiWriter([40, 80])
for i in range(batch_size):
    midiWriter.make_midi_using_onsets(generated_midi[i,:,0].detach().cpu().numpy(), generated_midi[i,:,3].detach().cpu().numpy(), emotion+"184_gpTime"+str(i), 0.05)
# #we're gonna save to midi files
#

