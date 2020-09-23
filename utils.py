import numpy as np
import torch
import torch.nn as nn
import math
import os
import scipy.signal
from torch.utils.data import Dataset, random_split, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import midiutil
import json

#for now we will forget about conditioning, just try to generate some reasonable midi ok

class MidiWriter():
    def __init__(self, midi_range):
        self.midi_range = midi_range

    def make_midi_using_onsets(self, encoded_pitches, onsets, file_name, timestep, is_encoded=True):
        if is_encoded:
            midi_pitches = (encoded_pitches + 1) / 2 * (self.midi_range[1] - self.midi_range[0]) + self.midi_range[0]
        else:
            midi_pitches = encoded_pitches
        print("original pitches ", midi_pitches)
        #first, we filter to figure out
        #diff = np.diff(midi_pitches)
        #big_diffs = np.where(abs(diff) > 3)
        #we're looking for segments that are like,
        #counterexamples: 62, 46, 40 45, 44, - the 46 is what we want


        zero_indices = np.where(midi_pitches <= self.midi_range[0]+5)[0]
        final_zero_indices = list(zero_indices)
        zero_segments = self.get_segments(zero_indices)
        for i, seg in enumerate(zero_segments):
            #explore to the left!
            if i != 0:
                lower_limit = zero_segments[i-1][1]
            else:
                lower_limit = -1
            check = seg[0] - 1
            last = seg[0]
            while check > lower_limit:
                if abs(midi_pitches[check] - midi_pitches[last]) < 1:
                    final_zero_indices.append(check)
                    last = check
                    check = check-1
                else:
                    break

            #explore to the right!
            for i, seg in enumerate(zero_segments):
                if i != len(zero_segments)-1: #if we're not at the end
                    upper_limit = zero_segments[i + 1][0]
                else:
                    upper_limit = len(midi_pitches)
                check = seg[1]
                last = seg[1]-1
                while check < upper_limit:
                    if abs(midi_pitches[check] - midi_pitches[last]) < 2:
                        final_zero_indices.append(check)
                        last = check
                        check = check + 1
                    else:
                        break

        final_zero_indices = np.unique(sorted(final_zero_indices))
        print("final_zero_indices ", final_zero_indices)
        mask = np.ones(len(midi_pitches))
        if len(final_zero_indices) > 0:
            mask[final_zero_indices] = 0
        midi_pitches = midi_pitches * mask
        midi_pitches = scipy.signal.medfilt(midi_pitches,3) * mask #have to mult by mask again, basically we're adding more zeros

        #print("final midi_pitches: ",midi_pitches)
        becomes_nonzero = np.array([thing[1] for thing in self.get_segments(final_zero_indices)])
        #print("becomes_nonzero ", becomes_nonzero)
        becomes_nonzero = list(becomes_nonzero[becomes_nonzero < len(midi_pitches)])

        onsets_new = onsets.copy()
        onsets_new[0] = 1 #force
        onset_indices = np.where(onsets_new > 0)[0]
        #print("initial onset_indices: ", onset_indices)
        onset_indices = np.unique(list(onset_indices) + becomes_nonzero)
        final_onset_indices = []
        for o in onset_indices:
            if midi_pitches[o] != 0:
                final_onset_indices.append(o)
            else:
                print("eliminating ", o)
        #print("final onset_indices: ", onset_indices)
        onset_times = np.array(final_onset_indices) * timestep
        print("onset times ", onset_times)
        print("midi pitches ", midi_pitches)
        self.make_midi(onset_times, midi_pitches, file_name, timestep)

    def get_segments(self, zero_indices):
        segments = []
        current_segment = []
        for i in range(len(zero_indices)):
            if i == (len(zero_indices) - 1):
                if len(current_segment) == 0:
                    segments.append([zero_indices[i], zero_indices[i] + 1])
                else:
                    current_segment.append(zero_indices[i]+1)
                    segments.append(current_segment)
            else:
                if len(current_segment) == 0:
                    if zero_indices[i + 1] != zero_indices[i] + 1:  ##if the next one is not a zero index, we're at the end
                        segments.append([zero_indices[i], zero_indices[i] + 1])
                    else:
                        current_segment.append(zero_indices[i])
                elif zero_indices[i + 1] != zero_indices[i] + 1:  # if the next one is not a zero index, we're at the end
                    current_segment.append(zero_indices[i]+1)
                    segments.append(current_segment)
                    current_segment = []

        return segments

    def make_midi_using_time_data(self, encoded_data, file_name, timestep, is_encoded = True):
        #assume encoded_data is timesteps x features
        if is_encoded:
            midi_pitches = (encoded_data[:,0] + 1) / 2 * (self.midi_range[1] - self.midi_range[0]) + self.midi_range[0]
        else:
            midi_pitches = encoded_data[:,0]

    def get_onsets_from_time_data(self, encoded_data, timestep):
        pass
        #onsets happen when midi changes to something nonzero


    def get_onsets(self, tuned_midi, timestep):
        """
        Input: what your model outputs. But this is written with parameters for 0.01 step size
        Output: onset times in seconds
        """
        non_note_value = 0
        time = np.arange(len(tuned_midi)) * timestep  # make this your own time array with timesteps

        pitch_deriv = np.diff(tuned_midi, prepend=0)
        pitch_derivs_pos = np.where(pitch_deriv < 0, 0, pitch_deriv)
        pitch_derivs_neg = np.abs(np.where(pitch_deriv > 0, 0, pitch_deriv))

        # 21 and 8 are in terms of samples (at .01 intervals) - would need to play with parameters for 0.1 intervals
        wlen_time = 0.21
        distance_time = 0.08
        wlen = int(np.round(wlen_time / timestep) + int(np.round(wlen_time / timestep) + 1) % 2)
        distance = int(np.ceil(distance_time / timestep))
        note_peaks_1 = scipy.signal.find_peaks(pitch_derivs_neg, prominence=0.1, wlen=wlen, distance=distance)[0]
        note_peaks_2 = scipy.signal.find_peaks(pitch_derivs_pos, prominence=0.1, wlen=wlen, distance=distance)[0]

        # get (most of) the potential onsets
        note_peaks = np.array(sorted(list(note_peaks_1) + list(note_peaks_2)))

        print("peaks: ", note_peaks)
        # Filter out some onsets
        pitch_window_time = 0.04
        pitch_window = int(np.round(pitch_window_time / timestep)) #will actually be 0 for us
        pitch_onsets = []
        for i in note_peaks:
            # check if there is a pitch for long enough after the onset for the note to count (4 samples -> 0.04 seconds)
            # the "-1" should deal with the fact that yours uses 1 instead of 0

            if np.count_nonzero(tuned_midi[i:min(i + pitch_window, len(tuned_midi))]) == pitch_window:
                pitch_onsets.append(time[i])
        # print("we counted ", pitch_onsets)
        # Make sure we have all onsets
        # We need to make super sure that we get onsets when midi goes from 1 to real pitch. They are called super onsets!
        super_onsets = []
        previous_pitch = non_note_value
        for i in range(len(tuned_midi)):
            if tuned_midi[i] != non_note_value and previous_pitch == non_note_value:
                super_onsets.append(time[i])
                if time[i] not in pitch_onsets:
                    pitch_onsets.append(time[i])
            previous_pitch = tuned_midi[i]

        pitch_onsets = sorted(pitch_onsets)


        # print("after super we have ", pitch_onsets)
        # now we go back through everything and make sure nothing is too close together
        # But make sure we always keep onsets on a 1-> real midi pitch
        # If you're fine with onsets being 0.1 apart (your interval size) you can delete all of this
        # I changed the code a bit and it's maybe not perfectly tested...
        # final_onsets = []
        # closeness_threshold = 0.08
        # already_processed_indices = []
        # for i, onset in enumerate(pitch_onsets):
        #     if i in already_processed_indices:
        #         continue
        #     already_processed_indices.append(i)
        #     suspicious_onset_indices = [i]
        #     for j in range(i, len(pitch_onsets) - 1):
        #         if abs(pitch_onsets[j + 1] - pitch_onsets[j]) < closeness_threshold:
        #             suspicious_onset_indices.append(j + 1)
        #             already_processed_indices.append(j + 1)
        #         else:
        #             break
        #
        #     found_super_onset = False
        #     if len(suspicious_onset_indices) > 1:
        #         for si in suspicious_onset_indices:
        #             time_index = abs((pitch_onsets[si] - time)).argmin()
        #             if pitch_onsets[si] in super_onsets:
        #                 # it's on a super onset, has to stay included
        #                 found_super_onset = True
        #                 final_onsets.append(pitch_onsets[si])
        #         if not found_super_onset:
        #             # prioritize the last onset found (just by default, pretty arbitrary)
        #             final_onsets.append(pitch_onsets[suspicious_onset_indices[-1]])
        #     else:  # all is well
        #         final_onsets.append(onset)
        #
        # return np.array(final_onsets)
        return np.array(pitch_onsets)

    def make_midi(self, onsets, midi, file_name, timestep):
        track = 0
        channel = 0
        bpm = 120

        def time_to_beat(time):
            return time / 60 * bpm

        onset_indices = (onsets / timestep).astype(np.int32)
        print("onset indices ", onset_indices)
        if len(onset_indices) > 0 and np.count_nonzero(midi) > 0:
            # make midi file template (bpm is just random at 120)
            myMidi = midiutil.MIDIFile(1)
            myMidi.addTempo(track, 0, bpm)
            myMidi.addTimeSignature(0, 0, 4, 2, 24)

            pitch_segments = []
            for i in range(len(onset_indices)):
                # find end of the note:
                # either when the frequency next goes to 0, or when the next onset is
                next_onset = len(midi) - 1
                next_0_index = len(midi) - 1
                if i < len(onset_indices) - 1:  # not the last one
                    next_onset = onset_indices[i + 1]
                for j in range(onset_indices[i] + 1, len(midi)):
                    if midi[j] == 0:
                        next_0_index = j
                        break
                #rint("onset_indices[i] ", onset_indices[i])
                #print("next onset ", next_onset)
                #print("next_0_index ", next_0_index)
                # pitches within this "midi note" (for pitch bends)
                if onset_indices[i] == len(midi) - 1:
                    segment = [midi[onset_indices[i]]]
                else:
                    segment = midi[onset_indices[i]:min(next_onset, next_0_index)]

                # average integer midi to use as center point for pitch bends
                average_midi = int(np.round(np.mean(segment)))

                # timing calculations
                start_beat = time_to_beat(onset_indices[i] * timestep)
                duration_beat = time_to_beat(len(segment) * timestep)
                myMidi.addNote(track, channel, average_midi, start_beat, duration_beat, 50)

                #print("original midi ", midi[onset_indices[i]:onset_indices[i]+len(segment)])

                # do a pitch bend for every sample according to tuned_midi curve
                mean_filtered_midi = self.moving_average(midi[onset_indices[i]:onset_indices[i]+len(segment)])
                #print("filtered ", mean_filtered_midi)
                for j in np.arange(len(segment)):
                    beat = start_beat + time_to_beat(j * timestep)
                    index = j + onset_indices[i]
                    # myMidi.addPitchWheelEvent(track, channel, beat,
                    #                           min(8192, max(-8192, int(8192 * (midi[index] - average_midi) / 2))))
                    myMidi.addPitchWheelEvent(track, channel, beat,
                                              min(8192, max(-8192, int(8192 * (mean_filtered_midi[j] - average_midi) / 2))))

            # write midi file
            with open(file_name + ".mid", "wb") as output_file:
                myMidi.writeFile(output_file)

    def moving_average(self, a, n=3):
        if len(a) == 1:
            return a
        else:
            to_use = np.concatenate((a, a[-1:], a[-1:]))
        ret = np.cumsum(to_use, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

class EmotionDataset(Dataset):
    def __init__(self, path_to_files, fps, device):
        self.device = device
        self.all_data = []
        self.emotion_embeddings = self.get_emotion_embeddings()
        self.midi_range = [45, 80]
        self.fps = fps
        # self.midiWriter = MidiWriter()
        for folder in os.listdir(path_to_files):
            if folder.startswith("Voice"):
                #print("folder ", folder)
                midi = np.load(path_to_files + folder + "/tuned_midi.npy")
                onsets_in_seconds = np.load(path_to_files + folder + "/onsets.npy")
                if np.count_nonzero(midi) == 0:
                    #print("ZERO MIDIS")
                    continue
                #print("midi ", midi)
                #print("onsets in seconds ", onsets_in_seconds)
                processed_midi, onsets_in_seconds = self.process_midi_and_onsets(midi, onsets_in_seconds)  # CHANGE

                if processed_midi is not None:
                    processed_midi = self.get_midi_resampled(processed_midi, self.fps)
                    processed_midi = processed_midi.reshape(-1, 1)

                    # processed_midi_pitches = np.ones(processed_midi.shape)
                    # processed_midi_pitches[processed_midi == 0] = 1 #turn 0s into 1s
                    # processed_midi = np.concatenate((processed_midi, processed_midi_pitches), axis=1)
                    emotion = folder.split("_")[2].lower()
                    embeddings = np.tile(self.emotion_embeddings[emotion], (processed_midi.shape[0], 1))

                    #onsets
                    onset_indices = np.round(onsets_in_seconds * self.fps).astype(int)
                    onset_indices = np.unique(onset_indices)
                    #print("onset_indices ", onset_indices)
                    index_to_cut = None
                    onset_indices = onset_indices[onset_indices < processed_midi.shape[0]]
                    #print("new onset_indices ", onset_indices)
                    #print("processed_midi.shape[0] ", processed_midi.shape[0])

                    onset_data = np.ones((processed_midi.shape[0], 1))*-1 #-1 for no onset, 1 for onset
                    onset_data[onset_indices,0] = 1
                    #print(onset_data)
                    data = np.concatenate((processed_midi, onset_data, embeddings), axis=1)
                    # targets = np.roll(data, -1, axis=0)[:, 0].reshape(-1, 1)
                    # targets[-1, :] = 1 #force the ending to always be a rest
                    # targets_pitches = np.ones(targets.shape)  # try adding the info of rests in a different way
                    # targets_pitches[targets == 0] = 0
                    # targets = np.concatenate((targets, targets_pitches), axis=1)
                    self.all_data.append(torch.Tensor(data))#, torch.Tensor(targets)))
        self.num_features = self.all_data[0].shape[1]

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return self.all_data[idx]

    def get_midi_resampled(self, midi, fps):
        #fps is frames per second - original is 100
        #we will check this later to make sure all is well
        assert(fps <=100), "fps must be 100 or lower"
        num_frames = int(100 / fps) #number of .01 frames to include
        #pad 0s until we're at a happy length
        padded_midi = np.pad(midi, (0,num_frames - len(midi)%num_frames),mode="constant",constant_values=-1)
        padded_midi = padded_midi.reshape((int(len(padded_midi)/num_frames), num_frames)) #I checked and this is fine by default

        final_midi = np.zeros(padded_midi.shape[0])
        for i in range(padded_midi.shape[0]):
            row = padded_midi[i,:]
            if np.count_nonzero(row+1) >= num_frames/2:
                #it's a pitch! we average the nonzero elements
                final_midi[i] = np.mean(row[row > -1])
            else:
                final_midi[i] = -1 #it's a rest

        return final_midi

    # def create_midi_file(self, encoded_pitches, fileName, fps = None):
    #     """
    #     encoded_pitches: pitches that would be directly output by generator (reshaped) - -1 to 1, rests are -1
    #     fps: frames per second the samples are, default to self.fps but can be changed
    #     """
    #     if fps == None:
    #         fps = self.fps
    #     timestep = 1/fps
    #     midi_pitches = (encoded_pitches+1)/2 * (self.midi_range[1] - self.midi_range[0]) + self.midi_range[0]
    #     midi_pitches[midi_pitches <= self.midi_range[0]] = 0 #rests
    #     print("midi pitches: ", midi_pitches)
    #     onsets = self.midiWriter.get_onsets(midi_pitches, timestep)
    #     print("onsets: ", onsets)
    #     self.midiWriter.make_midi(onsets, midi_pitches, fileName, timestep)

    def get_emotion_embeddings(self):
        emotions = {"interest": 0, "amusement": 1, "pride": 2, "joy": 3, "pleasure": 4, "contentment": 5, "love": 6,
                    "admiration": 7, "relief": 8, "compassion": 9, "sadness": 10, "guilt": 11, "regret": 12,
                    "shame": 13, "disappointment": 14, "fear": 15, "disgust": 16, "contempt": 17, "hate": 18,
                    "anger": 19}
        emotion_embeddings = {}
        for emotion in emotions:
            emotion_index = emotions[emotion]
            emotion_angle = (18 * emotion_index + 9) * np.pi / 180
            quadrant = emotion_index // 5
            quadrant_angle = quadrant * np.pi / 2 + np.pi / 4
            # now we have to normalize the data to be between 0 and 1
            emotion_embeddings[emotion] = (np.array(
                [np.cos(emotion_angle), np.sin(emotion_angle), np.cos(quadrant_angle), np.sin(quadrant_angle)]) + 1) / 2
        return emotion_embeddings

    def process_midi(self, midi):
        # delete 0s at the beginning and end, add 10 at the end. You can keep them at the end, normalize between -1 and 1
        new_midi = np.clip((midi - self.midi_range[0]) / (self.midi_range[1] - self.midi_range[0]), 0, 1)
        nonzeros = np.where(new_midi > 0)[0]
        new_midi = new_midi * 2 - 1
        if len(nonzeros) == 0:
            return None
        first_nonzero = nonzeros[0]
        last_nonzero = nonzeros[-1]
        # new_midi = np.concatenate((new_midi, -np.ones(10)))  # add 10 rests at the end, trigger end later
        return new_midi[first_nonzero:last_nonzero]

    def process_midi_and_onsets(self, midi, onsets):
        # delete 0s at the beginning and end, add 10 at the end. You can keep them at the end, normalize between -1 and 1
        new_midi = np.clip((midi - self.midi_range[0]) / (self.midi_range[1] - self.midi_range[0]), 0, 1)
        nonzeros = np.where(new_midi > 0)[0]
        new_midi = new_midi * 2 - 1
        if len(nonzeros) == 0:
            return None
        first_nonzero = nonzeros[0]
        last_nonzero = nonzeros[-1]

        onsets_new = onsets - first_nonzero * 0.01
        onsets_new = onsets_new[onsets_new >= 0]
        # new_midi = np.concatenate((new_midi, -np.ones(10)))  # add 10 rests at the end, trigger end later
        return new_midi[first_nonzero:last_nonzero], onsets_new

    def get_data_with_noise(self, batched_input_data):#, batched_target_data):
        # note: this assumes 0 is mask value
        # let's start by just actually shifting things randomly
        batched_input = batched_input_data.clone().detach()
        # batched_target = batched_target_data.clone().detach()
        pitch_shift_range = [-5 / (self.midi_range[1] - self.midi_range[0])*2,
                             5 / (self.midi_range[1] - self.midi_range[0])*2]
        for batch in range(batched_input.size()[0]):
            # print("original: ", np.array(batched_input[:,batch,0]))

            # print("batched_input size ", batched_input.size())
            # print("where ", np.where(batched_input[:,batch,0]==0))
            rests = np.where(batched_input[batch, :, 0] == -1)[0]
            shift_value = np.random.random() * (pitch_shift_range[1] - pitch_shift_range[0]) + pitch_shift_range[0]
            batched_input[batch, :, 0] += shift_value
            batched_input[batch,rests,  0] = -1
            batched_input[batch, :,  0] = torch.clamp(batched_input[batch, :, 0], 0, 1).to(self.device)
            # print("new ", np.array(batched_input[:,batch,0]))

            # zeros = np.where(batched_target[batch, :, 0] == 0)[0]
            # batched_target[batch,:,  0] += shift_value
            # batched_target[batch, zeros, 0] = 0
            # batched_target[batch, :, 0] = torch.clamp(batched_target[batch, :, 0], 0, 1).to(device)

        return batched_input#, batched_target

class MidiWriterTiming():
    def __init__(self, midi_range):
        self.midi_range = midi_range

    def make_midi(self, raw_input_data):
        pass
        #rules: let's check the source code ugh.

class EmotionDatasetTiming(Dataset):
    def __init__(self, path_to_files, fps, device):
        #by 5PM have all your code ready so you can train curly

        #we want our output data to be midi pitch, duration of the note, tuning amount, and time left
        #at 0.05 intervals instead
        self.reference_folder = "../prosodyProject/processed_output_edited"
        self.device = device
        self.all_data = []
        self.normalize_min = -1
        self.normalize_max = 1  # we're gonna do this instead of -1 to 1 for now, I have a dream
        self.emotion_embeddings = self.get_emotion_embeddings()
        self.midi_range = [40, 80] #this should give enough wiggle room at the bottom for rests and offsets
        self.fps = fps
        self.old_frames_in_new = int(100/self.fps)

        self.max_time_seconds = 10 #longest note

        for folder in os.listdir(self.reference_folder):
            if folder.startswith("Voice"):
                #print("folder ", folder)
                try:
                    with open(path_to_files + folder + "_notes.json") as my_file:
                        midi = json.load(my_file)
                except:
                    print("folder not found: ", folder)
                    continue

                if midi is not None:
                    first_time = midi[0]["start"]
                    for i in range(len(midi)):
                        midi[i]["start"] = midi[i]["start"] - first_time
                    emotion = folder.split("_")[2].lower()

                    onsets_original_fps = np.array([m["start"] for m in midi])
                    onsets_in_seconds = np.array([m["start"]*0.01 for m in midi])
                    onsets_init = np.round(onsets_in_seconds * self.fps).astype(int)
                    #get unique onsets and their indices for later - default to taking the second one, it's probs longer
                    onset_indices = []
                    onsets = []
                    if len(onsets_init) > 1:
                        for i in range(0,len(onsets_init)-1):
                            if onsets_init[i] == onsets_init[i-1]:
                                pass
                            else:
                                onsets.append(onsets_init[i])
                                onset_indices.append(i)
                        onsets.append(onsets_init[-1]) #always take the last one
                        onset_indices.append(len(onsets_init)-1)
                    else:
                        onsets = [onsets_init[0]]
                        onset_indices = [0]
                    onsets = np.array(onsets)
                    if len(onsets_init) != len(onsets):
                        pass

                    with open(path_to_files + folder + "_pitchbend.json") as my_file:
                        pitchbend_loaded = json.load(my_file)

                    #set pitchbend_loaded to start at time 0
                    for i in range(len(pitchbend_loaded)):
                        pitchbend_loaded[i][0] = pitchbend_loaded[i][0] - first_time

                    #total frames for our final array
                    #print("pitchbend_loaded[-1][0] ", pitchbend_loaded[-1][0])
                    total_frames = int(np.ceil((pitchbend_loaded[-1][0]+1) * 0.01 * self.fps))
                    #print("onsets first ", onsets)
                    #print("total_frames: ", total_frames)
                    onset_where = np.array(np.where(onsets < total_frames)[0]).astype(int)
                    #print("onset_where ", onset_where)
                    onsets = onsets[onset_where]
                    #print("onset_indices at first ", onset_indices)
                    #print("onset_where yet again ", onset_where)
                    onset_indices = np.array(onset_indices)[onset_where]
                    onsets_one_hot = np.zeros(total_frames)

                    #print("onsets: ", onsets)

                    onsets_one_hot[onsets] = 1
                    #will be array at original 100fps sample rate
                    pitchbends_loaded_fps = np.zeros(pitchbend_loaded[-1][0]+1)

                    #fill it in with the pitchbends
                    for i in range(len(pitchbend_loaded)):
                        pitchbends_loaded_fps[pitchbend_loaded[i][0]] = pitchbend_loaded[i][1]

                    #now create the final array
                    pitchbends = np.zeros(total_frames)
                    for i in range(len(pitchbends)):
                        bends = pitchbends_loaded_fps[i*self.old_frames_in_new:i*self.old_frames_in_new + self.old_frames_in_new]
                        nonzero_bends = bends[np.nonzero(bends)[0]]
                        if len(nonzero_bends) > 0:
                            pitchbends[i] = np.mean(nonzero_bends)

                    embeddings = np.tile(self.emotion_embeddings[emotion], (total_frames, 1))

                    raw_output = np.zeros((1, total_frames,4+4)) #4 emotion dim, 4 features. This is un-normalized
                    raw_output[0,:,4:] = embeddings

                    #first we do everything except pitchbend
                    for index, frame in enumerate(onsets):
                        index_for_midi = onset_indices[index]
                        if index == len(onsets)-1: #if it's the last onset
                            last_frame = total_frames
                        else:
                            last_frame = min(onsets[index+1], frame + int(np.round(midi[index_for_midi]["duration"]*0.01 * self.fps)))
                        #pitches
                        #raw_output[0,frame:last_frame,0] = midi[index_for_midi]["pitch"] #indexing should still be fine like this
                        raw_output[0, frame:last_frame, 0] = midi[index_for_midi]["pitch"] + pitchbends[frame:last_frame]
                        #duration in seconds
                        raw_output[0, frame:last_frame, 1] = midi[index_for_midi]["duration"]*0.01
                        #time since last onset in seconds
                        raw_output[0, frame:last_frame, 2] = np.arange(last_frame-frame) / self.fps
                        #onsets
                        raw_output[0, frame:last_frame, 3] = onsets_one_hot[frame:last_frame]

                        #pitch bends
                        #raw_output[0,frame:last_frame, 3] = pitchbends[frame:last_frame] #just in case there was a weird thing elsewhere

                    data = self.normalize_data(raw_output)
                    #print(data)

                    self.all_data.append(torch.Tensor(data[0,:,:]))#, torch.Tensor(targets)))
        self.num_features = self.all_data[0].shape[1]

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return self.all_data[idx]

    def get_emotion_embeddings(self):
        emotions = {"interest": 0, "amusement": 1, "pride": 2, "joy": 3, "pleasure": 4, "contentment": 5, "love": 6,
                    "admiration": 7, "relief": 8, "compassion": 9, "sadness": 10, "guilt": 11, "regret": 12,
                    "shame": 13, "disappointment": 14, "fear": 15, "disgust": 16, "contempt": 17, "hate": 18,
                    "anger": 19}
        emotion_embeddings = {}
        for emotion in emotions:
            emotion_index = emotions[emotion]
            emotion_angle = (18 * emotion_index + 9) * np.pi / 180
            quadrant = emotion_index // 5
            quadrant_angle = quadrant * np.pi / 2 + np.pi / 4
            # now we have to normalize the data to be between min and max
            emotion_embeddings[emotion] = (np.array(
                [np.cos(emotion_angle), np.sin(emotion_angle), np.cos(quadrant_angle), np.sin(quadrant_angle)]) + 1) / 2 * (self.normalize_max-self.normalize_min) + self.normalize_min
        return emotion_embeddings

    def normalize_data(self, raw_output):

        #pitch, duration, time since last onset, pitch bend
        # print("raw_output: ")
        # print(raw_output)
        # print("\n")
        new_midi = np.zeros(raw_output.shape)
        #first let's get everything from 0 to 1, then readjust (if needed)
        new_midi[0,:,0] = np.clip((raw_output[0,:,0] - self.midi_range[0]) / (self.midi_range[1] - self.midi_range[0]), 0, 1)
        new_midi[0,:,1] = np.clip(raw_output[0,:,1]/self.max_time_seconds, 0, 1)
        new_midi[0,:,2] = np.clip(raw_output[0,:,2]/self.max_time_seconds, 0, 1)
        #new_midi[0, :, 3] = np.clip((raw_output[0,:,3] + 2) / 4, 0, 1) #-2 to 2 is the range for pitch bend
        new_midi[0,:,3] = raw_output[0,:,3]

        new_midi[0,:,:] = new_midi[0,:,:] * (self.normalize_max-self.normalize_min) + self.normalize_min

        new_midi[:,:,4:] = raw_output[:,:,4:] #emotions

        return new_midi

    def get_data_with_noise(self, batched_input_data):#, batched_target_data):
        # note: this assumes 0 is mask value
        # let's start by just actually shifting things randomly
        semitone_shift = 1 #was 5
        batched_input = batched_input_data.clone().detach()
        pitch_shift_range = [-semitone_shift / (self.midi_range[1] - self.midi_range[0])*(self.normalize_max-self.normalize_min),
                             semitone_shift / (self.midi_range[1] - self.midi_range[0])*(self.normalize_max-self.normalize_min)]

        for batch in range(batched_input.size()[0]):
            # print("original: ", np.array(batched_input[:,batch,0]))

            # print("batched_input size ", batched_input.size())
            # print("where ", np.where(batched_input[:,batch,0]==0))
            rests = np.where(batched_input[batch, :, 0] == self.normalize_min)[0] #keep these as rests
            shift_value = np.random.random() * (pitch_shift_range[1] - pitch_shift_range[0]) + pitch_shift_range[0]
            batched_input[batch, :, 0] += shift_value
            batched_input[batch,rests,  0] = self.normalize_min
            batched_input[batch, :,  0] = torch.clamp(batched_input[batch, :, 0], self.normalize_min, self.normalize_max).to(self.device)
            # print("new ", np.array(batched_input[:,batch,0]))

            # zeros = np.where(batched_target[batch, :, 0] == 0)[0]
            # batched_target[batch,:,  0] += shift_value
            # batched_target[batch, zeros, 0] = 0
            # batched_target[batch, :, 0] = torch.clamp(batched_target[batch, :, 0], 0, 1).to(device)

        return batched_input

if __name__ == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    #dataset = EmotionDatasetTiming("../prosodyProject/processed_midi_info_edited/", 20,device)
    m = MidiWriter([40,80])
    test_midi = np.array([45.5,45,42,40,46,65,63,87,88,88,89,40,44.6,45.5])
    test_onsets = np.ones(len(test_midi))*-1
    test_onsets[7] = 1
    m.make_midi_using_onsets(test_midi,test_onsets,"none",0.05,is_encoded=False)