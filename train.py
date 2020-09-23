import numpy as np
import torch
import torch.nn as nn
import math
import os
import scipy.signal
from torch.utils.data import Dataset, random_split, DataLoader
from utils import EmotionDataset, MidiWriter, EmotionDatasetTiming
from model import Generator, Discriminator, Generator2, Baseline_Model, Classifier
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

def pad_collate(batch):
    #batch is just a list of midis (and their gt concatted)
    x_lens = torch.Tensor([len(x) for x in batch])
    # print("batch size ", [b.size() for b in batch])
    x_pad = pad_sequence(batch, padding_value=0, batch_first=True)
    # print("returning from pad_collate")
    return x_pad, x_lens

def train(batch_size, num_epochs, save_every, device):
    dataset = EmotionDataset("../prosodyProject/processed_output_edited/", 10)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=pad_collate,drop_last=True)
    generator_input_channels = 5 + 4 + 1 #we'll try noise vector of 6?
    discriminator_input_channels = 4 + 1 + 1 + 1#emotions, onsets, pitches, average onsets/time
    generator = Generator(generator_input_channels, batch_size, device)
    generator.to(device)
    discriminator = Discriminator(discriminator_input_channels, batch_size, device)
    discriminator.to(device)

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0025)
    discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=0.004)

    loss = nn.BCELoss()

    last_disc_loss = -1
    last_gen_loss = -1

    for epoch in range(num_epochs):
        for (x_padded, x_lens) in loader:
            # print("x_lens ", x_lens)
            this_batch_size = x_padded.size()[0]
            generator_optimizer.zero_grad()

            #real data
            orig_labels = torch.ones((this_batch_size, 1)).to(device)
            label_noise = (torch.rand(orig_labels.size()).to(device)-0.5)*0.2
            true_labels = orig_labels + label_noise
            true_data = dataset.get_data_with_noise(x_padded).to(device)
            #get the rest numbers
            avg_true_onsets = torch.sum(torch.nn.functional.relu(true_data[:,:,1]), axis=1)*1/x_lens.to(device)
            avg_true_onsets = avg_true_onsets.view(-1, 1, 1)
            avg_true_onsets = avg_true_onsets.repeat(1,x_padded.size()[1],1)
            #
            # avg_true_onsets = torch.ones(x_padded.size()[0], x_padded.size()[1], 1)
            # avg_true_onsets[:,:,0] = avg_true_onsets_temp
            #print("avg_true_onsets: ", avg_true_onsets)
            #print("avg_true_onsets size: ", avg_true_onsets.size())
            true_data = torch.cat((true_data, avg_true_onsets), axis=2)
            #print("true data size ", true_data.size())
            #print("and then ", true_data[:,:,2:-1].size())
            noise = torch.randn(x_padded.size()[0], x_padded.size()[1], generator_input_channels).to(device) #gaussian noise my man
            #now change to be the same emotion embedding as true_data
            noise[:,:,0:5] = true_data[:,:,2:]
            # print("noise[0,5:15,:] ", noise[0, 5:15, :])
            # print("noise[1,5:15,:] ", noise[1, 5:15, :])
            #noise = pack_padded_sequence(noise, x_lens, batch_first = True, enforce_sorted=False)

            generated_data = generator(noise, x_lens)

            # print("generated_data size ", generated_data.size())
            avg_gen_onsets = torch.sum(torch.nn.functional.relu(generated_data[:, :, 1]), axis=1) * 1 / x_lens.to(device)
            avg_gen_onsets = avg_gen_onsets.view(-1, 1, 1)
            avg_gen_onsets = avg_gen_onsets.repeat(1, x_padded.size()[1], 1)
            #print("avg_gen_onsets: ", avg_gen_onsets)
            final_padded_gen = torch.cat((generated_data, true_data[:,:,2:-1], avg_gen_onsets), axis = 2)
            # print("final_padded_gen size ", final_padded_gen.size())
            # print("final_padded_gen[0,10:15,:] ", final_padded_gen[0,10:15,:])
            # print("final_padded_gen[0,-5:,:] ", final_padded_gen[0,-5:,:])
            # print("final_padded_gen[1,-5:,:] ", final_padded_gen[1,-5:,:])
            #print("size of final_padded_gen ", final_padded_gen.size())
            #train generator (NOT discriminator)
            #invert labels!
            #if disc. thinks the false things are true, then loss is small
            #if disc. thinks the false things are false, loss is high
            #have to concatenate the emotions. Buuuuut we have a packed sequence cryyyyy
            generated_data = pack_padded_sequence(final_padded_gen, x_lens, batch_first = True, enforce_sorted=False)
            generator_discriminator_out = discriminator(generated_data)
            generator_loss = loss(generator_discriminator_out, true_labels)
            if last_gen_loss > 0.7*last_disc_loss or last_gen_loss == -1:
                generator_loss.backward()
                generator_optimizer.step()
                last_gen_loss = float(generator_loss)

            #discriminator!
            discriminator_optimizer.zero_grad()
            packed_true_data = pack_padded_sequence(true_data, x_lens, batch_first=True, enforce_sorted=False)
            true_discriminator_out = discriminator(packed_true_data)
            #print("disc thought true data was ", torch.mean(true_discriminator_out))
            true_discriminator_loss = loss(true_discriminator_out, true_labels)

            #detatch so we don't accidentally train the generator too
            detached_final_padded_gen = final_padded_gen.detach()
            generated_data = pack_padded_sequence(detached_final_padded_gen, x_lens, batch_first=True, enforce_sorted=False)

            generator_discriminator_out = discriminator(generated_data)
            #print("disc thought fake data was ", torch.mean(generator_discriminator_out))
            generator_discriminator_loss = loss(generator_discriminator_out, torch.zeros((this_batch_size,1)).to(device))

            discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
            if last_disc_loss > 0.7*last_gen_loss or last_disc_loss == -1:
                discriminator_loss.backward()
                discriminator_optimizer.step()
                last_disc_loss = float(discriminator_loss)

        if epoch% save_every == 0:
            print("saving, generator loss: ", generator_loss, " disc loss ", discriminator_loss)
            torch.save({"epoch": epoch, "model_state_dict": generator.state_dict(), "optimizer_state_dict": generator_optimizer.state_dict(),
                 "loss": generator_loss}, "checkpoints/generator_checkpoint"+str(epoch)+".pth.tar")
            torch.save({"epoch": epoch, "model_state_dict": discriminator.state_dict(),
                        "optimizer_state_dict": discriminator_optimizer.state_dict(),
                        "loss": discriminator_loss}, "checkpoints/discriminator_checkpoint"+str(epoch)+".pth.tar")

class Trainer_Classifier():
    def __init__(self, batch_size, device, index_to_start_conditioning):
        #need to do train/test/val split though girl
        self.batch_size = batch_size
        self.device = device
        self.dataset = EmotionDatasetTiming("../prosodyProject/processed_midi_info_edited/", 20, self.device)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,
                                 collate_fn=pad_collate, drop_last=True)
        self.classifier = Classifier(4, batch_size, device, 4,hidden_size=50, num_layers=3) #not 4+4 because no emotions given
        self.classifier.to(device)
        self.index_to_start_conditioning = index_to_start_conditioning #probably 4
        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.0001)
        self.losses = []
        self.bceloss = nn.BCELoss()
        self.crossentropy = nn.CrossEntropyLoss()

    def train(self,num_epochs, save_every):
        for epoch in range(num_epochs):
            for (x_padded, x_lens) in self.loader:
                real_data = self.dataset.get_data_with_noise(x_padded).to(self.device) #just leave as-is for now, simpler
                train_data = pack_padded_sequence(real_data[:,:,:self.index_to_start_conditioning], x_lens, batch_first=True, enforce_sorted=False)
                emotion_info = real_data[:,0,self.index_to_start_conditioning:]
                output = self.classifier(train_data)
                loss = self.loss_quadrant(output, emotion_info)
                loss.backward()
                self.classifier_optimizer.step()
                self.losses.append(loss.item())
            print("last output ", output)
            print("loss: ", self.losses[-1])
            if epoch % save_every == 0:
                pass
                #torch.save({"epoch": epoch, "model_state_dict": self.classifier.state_dict(),
                 #           "optimizer_state_dict": self.classifier_optimizer.state_dict(),
                    #        "loss": self.losses[-1]},
                     #      "checkpoints_classifier1/classifier_checkpoint" + str(epoch) + ".pth.tar")

    def loss_2(self, output, emotion_info):
        #print("output: ", output)
        #print("compare to: ", emotion_info[:,0:2])
        emotion_loss = torch.dist(output, emotion_info[:,0:2])
        #print("emotion_loss: ", emotion_loss)
        return emotion_loss

    def loss_quadrant(self, output, emotion_info):
        quadrant_gt = torch.zeros(emotion_info.size(0),dtype=torch.long).to(self.device)
        for i in range(emotion_info.size(0)):
            if emotion_info[i,2] > 0:
                if emotion_info[i,3] > 0:
                    quadrant = 0
                else:
                    quadrant = 3
            else:
                if emotion_info[i, 3] > 0:
                    quadrant = 1
                else:
                    quadrant = 2
            quadrant_gt[i] = quadrant
        #print("quadrant_gt ", quadrant_gt)
        return self.crossentropy(output, quadrant_gt)



class Trainer_WGAN_GP():
    def __init__(self, batch_size, device, index_to_start_conditioning, num_critic=5, gp_weight=10):
        self.batch_size = batch_size
        self.device = device
        self.index_to_start_conditioning = index_to_start_conditioning
        self.gp_weight = gp_weight
        self.num_critic = num_critic

        self.dataset = EmotionDatasetTiming("../prosodyProject/processed_midi_info_edited/", 20, self.device)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=pad_collate,drop_last=True)
        self.generator_input_channels = 4 + 4   # we'll try noise vector of 4, and then 4 for emotions
        self.discriminator_input_channels = 4 + 4  # emotions, and our 4 other things (pitch, duration, time left, tuning)

        self.generator = Generator2(self.generator_input_channels, self.batch_size, self.device, output_dim=4)
        self.generator.to(device)
        self.discriminator = Discriminator(self.discriminator_input_channels, self.batch_size, self.device)
        self.discriminator.to(device)

        self.baseline = Baseline_Model(4, self.batch_size, self.device, output_dim=4)
        self.baseline.to(device)

        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0001, betas = (0.5, 0.999))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001, betas = (0.5, 0.999))
        self.baseline_optimizer = torch.optim.Adam(self.baseline.parameters(), lr=0.0001, betas=(0.5, 0.999))

        self.processing_on_gen_output = None

        self.num_steps = 0
        self.losses = {"disc": [], "gen": [], "gp": [], "baseline": []}


    def _critic_train_iteration(self, real_data, x_lens):
        #index_to_start_conditioning was 2 before
        generated_data = self._sample_generator(real_data, x_lens)
        #print("generated_data.size()", generated_data.size())
        packed_generated_data = pack_padded_sequence(generated_data, x_lens, batch_first=True, enforce_sorted=False)
        d_generated = self.discriminator(packed_generated_data)

        packed_real_data = pack_padded_sequence(real_data, x_lens, batch_first=True, enforce_sorted=False)
        d_real = self.discriminator(packed_real_data)

        gradient_penalty = self._compute_gradient_penalty(real_data, generated_data, x_lens)

        self.discriminator_optimizer.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        #d_loss = gradient_penalty
        d_loss.backward()
        self.discriminator_optimizer.step()
        self.losses["disc"].append(d_loss.item())


    def _sample_generator(self, real_data, x_lens, targets = None):
        noise = torch.randn(real_data.size()[0], real_data.size()[1], self.generator_input_channels).to(device)
        #noise first, then emotions
        noise[:, :, self.generator_input_channels-4:] = real_data[:, :, self.index_to_start_conditioning:]
        generated_data = self.generator(noise, x_lens, targets = targets)
        # print("generated_data size ", generated_data.size())
        # print("pitches gen: ", generated_data[0, :, 0])
        # print("durations gen: ", generated_data[0, :, 1])
        # print("time since onset gen: ", generated_data[0, :, 2])
        # print("onsets gen: ", generated_data[0, :, 3])
        # print("real_data[:, :, self.index_to_start_conditioning:]: ",real_data[:, :, self.index_to_start_conditioning:].size())
        final_padded_gen = torch.cat((generated_data, real_data[:, :, self.index_to_start_conditioning:]), axis=2)
        if self.processing_on_gen_output is not None:
            final_padded_gen = self.processing_on_gen_output(final_padded_gen, x_lens)
        return final_padded_gen #unpacked

    def _generator_train_iteration(self, real_data, x_lens):
        self.generator_optimizer.zero_grad()
        generated_data = self._sample_generator(real_data,x_lens)
        packed_generated_data = pack_padded_sequence(generated_data, x_lens, batch_first=True, enforce_sorted=False)
        d_generated = self.discriminator(packed_generated_data)
        g_loss = -d_generated.mean()
        g_loss.backward()
        self.generator_optimizer.step()
        self.losses["gen"].append(g_loss.item())

    def _compute_gradient_penalty(self, real_data, generated_data, x_lens):
        if True:
            alpha = torch.rand(self.batch_size, 1, 1) #i think
            alpha = alpha.expand_as(real_data).to(self.device)
            interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
            #interpolated = alpha * real_data.detach() + (1 - alpha) * generated_data.detach()
            interpolated.to(self.device)

            interpolated.requires_grad = True
            packed_interpolated = pack_padded_sequence(interpolated, x_lens, batch_first=True, enforce_sorted=False)

            with torch.backends.cudnn.flags(enabled=False):
                d_interpolated = self.discriminator(packed_interpolated)

                gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated,
                                       grad_outputs=torch.ones(d_interpolated.size()).to(self.device),
                                       create_graph=True, retain_graph=True)[0]
                gradients = gradients.view(self.batch_size, -1)
                gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
                gp_loss = self.gp_weight * ((gradients_norm - 1) ** 2).mean()
                #print("gp_loss ", gp_loss)
                self.losses["gp"].append(gp_loss.item())
            return gp_loss

    def train(self, num_epochs, save_every):
        for epoch in range(num_epochs):
            for (x_padded, x_lens) in self.loader:
                real_data = self.dataset.get_data_with_noise(x_padded).to(self.device) #just leave as-is for now, simpler
                self._critic_train_iteration(real_data, x_lens)
                self.num_steps += 1
                if self.num_steps % self.num_critic:
                    #train the generator
                    self._generator_train_iteration(real_data, x_lens)

            print("epoch ", epoch, " generator loss: ", self.losses["gen"][-1], " disc loss ", self.losses["disc"][-1], " gp loss ", self.losses["gp"][-1])
            if epoch % save_every == 0:
                print("saving, generator loss: ", self.losses["gen"][-1], " disc loss ", self.losses["disc"][-1], " gp loss ", self.losses["gp"][-1])
                #torch.save({"epoch": epoch, "model_state_dict": self.generator.state_dict(),
                            #"optimizer_state_dict": self.generator_optimizer.state_dict(),
                           # "loss": self.losses["gen"][-1]}, "checkpoints_gpTime/generator_checkpoint" + str(epoch) + ".pth.tar")
                #torch.save({"epoch": epoch, "model_state_dict": self.discriminator.state_dict(),
                            #"optimizer_state_dict": self.discriminator_optimizer.state_dict(),
                            #"loss": self.losses["disc"][-1]},
                           #"checkpoints_gpTime/discriminator_checkpoint" + str(epoch) + ".pth.tar")

    def pretrain(self, num_epochs):
        num_features = 4 #assume the features are the first things in the data
        for epoch in range(num_epochs):
            print("pretrain epoch ", epoch)
            my_counter = 0
            for (x_padded, x_lens) in self.loader:
                #print("original x_lens shape ", x_lens.size())
                real_data = self.dataset.get_data_with_noise(x_padded).to(self.device) #just leave as-is for now, simpler

                target_output = real_data[:,:,0:num_features].to(self.device) #index of where midi info is, change if plan changes

                self.generator_optimizer.zero_grad()

                for param in self.generator.parameters():
                    pass

                generated_data = self._sample_generator(real_data, x_lens, targets = target_output)[:,:,:num_features]

                aranged = np.vstack([np.arange(generated_data.size(1)) for i in range(generated_data.size(0))])
                ref_lengths = np.expand_dims(x_lens,1)
                mask = aranged < ref_lengths
                mask = torch.from_numpy(np.stack([mask for i in range(num_features)], axis = 2).astype(int)).to(self.device)

                from_array = generated_data * mask
                tar_array = target_output * mask

                g_loss = torch.dist(from_array,tar_array)
                g_loss.backward()

                for param in self.generator.parameters():
                    pass

                self.generator_optimizer.step()
                #print(g_loss.item())
                self.losses["gen"].append(g_loss.item())
                my_counter += 1

    def train_baseline(self, num_epochs, save_every):
        print("BASELINE")
        num_features = 4 #assume the features are the first things in the data
        for epoch in range(num_epochs):

            my_counter = 0
            for (x_padded, x_lens) in self.loader:
                #print("original x_lens shape ", x_lens.size())
                real_data = self.dataset.get_data_with_noise(x_padded).to(self.device) #just leave as-is for now, simpler

                target_output = real_data[:,:,0:num_features].to(self.device) #index of where midi info is, change if plan changes

                self.baseline.zero_grad()
                generated_data = self.baseline(real_data, x_lens)

                aranged = np.vstack([np.arange(generated_data.size(1)) for i in range(generated_data.size(0))])
                ref_lengths = np.expand_dims(x_lens,1)
                mask = aranged < ref_lengths
                mask = torch.from_numpy(np.stack([mask for i in range(num_features)], axis = 2).astype(int)).to(self.device)

                from_array = generated_data * mask
                tar_array = target_output * mask

                baseline_loss = torch.dist(from_array,tar_array)
                baseline_loss.backward()

                self.baseline_optimizer.step()
                self.losses["baseline"].append(baseline_loss.item())
                my_counter += 1
            print("epoch ", epoch, " baseline loss: ", self.losses["baseline"][-1])
            if epoch % save_every == 0:
                print("saving")
                #torch.save({"epoch": epoch, "model_state_dict": self.baseline.state_dict(),
                          #  "optimizer_state_dict": self.baseline_optimizer.state_dict(),
                          #  "loss": self.losses["baseline"][-1]},
                          # "checkpoints_baseline/baseline_checkpoint" + str(epoch) + ".pth.tar")

def train_classifier():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)
    trainer = Trainer_Classifier(32, device, 4)
    trainer.train(300, 10)

def train_gan():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)
    trainer = Trainer_WGAN_GP(32,device,4)
    trainer.train_baseline(300, 2)
    print("onto training")
    trainer.train(500,2)


if __name__ == "__main__":
    # dataset = EmotionDataset("../prosodyProject/processed_output_edited/", 20, "cpu")
    # midi= dataset.__getitem__(8)
    # print("shape: ", midi.shape)
    # midiWriter = MidiWriter([45,80])
    # midiWriter.make_midi_using_onsets(midi[:,0].numpy(), midi[:,1].numpy(), "testingMidi", 0.05)
    train_classifier()




