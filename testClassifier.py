import numpy as np
import torch
import torch.nn as nn
import math
import os
from torch.utils.data import Dataset, random_split, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from model import Classifier, Generator2
from utils import EmotionDataset, MidiWriter, EmotionDatasetTiming

def pad_collate(batch):
    #batch is just a list of midis (and their gt concatted)
    x_lens = torch.Tensor([len(x) for x in batch])
    # print("batch size ", [b.size() for b in batch])
    x_pad = pad_sequence(batch, padding_value=0, batch_first=True)
    # print("returning from pad_collate")
    return x_pad, x_lens
def num_correct(output, emotion_info):
    num_correct = 0
    quadrant_gt = torch.zeros(emotion_info.size(0), dtype=torch.long).to(device)
    for i in range(emotion_info.size(0)):
        if emotion_info[i, 2] > 0:
            if emotion_info[i, 3] > 0:
                quadrant = 0
            else:
                quadrant = 3
        else:
            if emotion_info[i, 3] > 0:
                quadrant = 1
            else:
                quadrant = 2
        quadrant_gt[i] = quadrant
    for i in range(output.size(0)):
        selection = torch.argmax(output[i,:])
        if selection == quadrant_gt[i]:
            num_correct += 1
    return num_correct

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print("device: ", device)
generator_input_channels = 4 + 4
batch_size = 32
classifier = Classifier(4, batch_size, device, 4,hidden_size=50, num_layers=3).to(device)
checkpoint = torch.load("checkpoints_classifier1/classifier_checkpoint90.pth.tar") #54 wasn't bad I think
classifier.load_state_dict(checkpoint['model_state_dict'])
classifier.eval()
classifier.to(device)

test_model = Generator2(generator_input_channels, batch_size, device,output_dim=4).to(device)
checkpoint = torch.load("checkpoints_gpTime/generator_checkpoint184.pth.tar") #54 wasn't bad I think
test_model.load_state_dict(checkpoint['model_state_dict'])
test_model.eval()
test_model.to(device)

###TEST DATASET
dataset = EmotionDatasetTiming("../prosodyProject/processed_midi_info_edited/", 20,device)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                 collate_fn=pad_collate, drop_last=True)
crossentropy = nn.CrossEntropyLoss()
total_correct = 0
total = 0
for (x_padded, x_lens) in loader:
    real_data = dataset.get_data_with_noise(x_padded).to(device)  # just leave as-is for now, simpler
    train_data = pack_padded_sequence(real_data[:, :, :4], x_lens, batch_first=True,
                                      enforce_sorted=False)
    emotion_info = real_data[:, 0, 4:]
    output = classifier(train_data)
    correct = num_correct(output, emotion_info)
    total_correct += correct
    total += output.size(0)

print("correct: ", total_correct)
print("total: ", total)
print("percent: ", total_correct/total)


####TEST MODEL
total_correct = 0
total = 0
for emotion in dataset.emotion_embeddings:
    time = time = np.random.randint(1,10)*20
    noise = torch.randn(batch_size, time, 4).to(device)
    tiled_emotions = np.tile(dataset.emotion_embeddings[emotion], (noise.size()[1], 1))
    tiled_emotions = np.tile(np.expand_dims(tiled_emotions, 0), (batch_size, 1, 1))
    tiled_emotions = torch.from_numpy(tiled_emotions).float().to(device)
    initial_sequence = torch.cat((noise, tiled_emotions), axis=2).to(device)
    generated_midi = test_model(initial_sequence, [initial_sequence.size()[1]])
    output = classifier(generated_midi)
    emotion
    correct = num_correct(output, tiled_emotions[:,0,:])
    total_correct += correct
    total += output.size(0)
print("correct: ", total_correct)
print("total: ", total)
print("percent: ", total_correct/total)
# def loss_quadrant(output, emotion_info):
#     quadrant_gt = torch.zeros(emotion_info.size(0),dtype=torch.long).to(self.device)
#     for i in range(emotion_info.size(0)):
#         if emotion_info[i,2] > 0:
#             if emotion_info[i,3] > 0:
#                 quadrant = 0
#             else:
#                 quadrant = 3
#         else:
#             if emotion_info[i, 3] > 0:
#                 quadrant = 1
#             else:
#                 quadrant = 2
#         quadrant_gt[i] = quadrant
#     #print("quadrant_gt ", quadrant_gt)
#     return self.crossentropy(output, quadrant_gt)