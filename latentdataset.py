import h5py
import os
import torch
from torch.utils.data import Dataset
import torchaudio
from AudioLoader.utils import tsv2roll
import numpy as np
import glob


class LatentAudioDataset(Dataset):
    def __init__(self, h5_dir, audio_dir, chunk_duration_sec=27):
        # self.h5_files = [os.path.join(h5_dir, f) for f in os.listdir(h5_dir) if f.endswith('.hdf5')]
        self.h5_files = glob.glob(os.path.join(
            h5_dir, '**', '*.hdf5'), recursive=True)
        self.h5_dir = h5_dir
        self.audio_dir = audio_dir
        # this was done in the latent extract phase, by default 27
        self.chunk_duration_sec = chunk_duration_sec

        # List to store (file_path, chunk_id) tuples
        self.file_chunk_pairs = []
        for h5_file in self.h5_files:
            with h5py.File(h5_file, 'r') as f:
                # Exclude groups that are not chunks (e.g., metadata)
                chunk_ids = [key for key in f.keys() if key.isdigit()]
                for chunk_id in chunk_ids:
                    self.file_chunk_pairs.append((h5_file, chunk_id))

    def __len__(self):
        return len(self.file_chunk_pairs)

    def __getitem__(self, idx):
        h5_file, chunk_id = self.file_chunk_pairs[idx]
        with h5py.File(h5_file, 'r') as f:
            group = f[chunk_id]
            # Load latent variables. 0 is the chunk id
            dac_latents = group['dac_latents'][()]

        # ---- TARGET VARS ------
        # Get the relative path of h5_file with respect to h5_dir, to handle /<year> subdirs
        relative_h5_path = os.path.relpath(h5_file, self.h5_dir)
        audio_relative_path = relative_h5_path.replace('.hdf5', '.wav')
        tsv_relative_path = relative_h5_path.replace('.hdf5', '.tsv')
        audio_path = os.path.join(self.audio_dir, audio_relative_path)
        tsv_path = os.path.join(self.audio_dir, tsv_relative_path)

        # Check if audio and TSV files exist
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not os.path.exists(tsv_path):
            raise FileNotFoundError(f"TSV file not found: {tsv_path}")

        # Load the piano roll
        # h5_filename = os.path.basename(h5_file)
        # audio_filename = os.path.splitext(h5_filename)[0] + '.wav'  # Assuming .wav extension
        # audio_path = os.path.join(self.audio_dir, audio_filename)

        # Calculate the start and end times for the chunk
        # NOTE: seems like we should start from the top, i.e. no delay, this is also done in latent extract phase
        chunk_index = int(chunk_id)
        start_time = chunk_index * self.chunk_duration_sec
        end_time = start_time + self.chunk_duration_sec

        # Load the audio segment
        sample_rate = torchaudio.info(audio_path).sample_rate
        start_frame = int(start_time * sample_rate)
        dur_frame = int(self.chunk_duration_sec * sample_rate)
        waveform, sr = torchaudio.load(
            audio_path,
            frame_offset=start_frame,
            num_frames=dur_frame
        )
        if waveform.dim() == 2:
            # converting a stereo track into a mono track
            waveform = waveform.mean(0)
        audio_length = len(waveform)

        # Load the TSV file
        # tsv_filename = os.path.splitext(h5_filename)[0] + '.tsv'
        # tsv_path = os.path.join(self.audio_dir, tsv_filename)
        tsv_data = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)
        # Adjust the tsv data to the chunk's time frame
        chunk_tsv = tsv_data[
            (tsv_data[:, 0] >= start_time) & (tsv_data[:, 1] <= end_time)
        ]
        # Adjust the onset and offset times to be relative to the chunk
        chunk_tsv[:, 0] -= start_time
        chunk_tsv[:, 1] -= start_time
        hop_length = 2048  # NOTE: pulled from latent extraction code as well, to check
        piano_roll, _ = tsv2roll(
            chunk_tsv, audio_length, sr, hop_length, max_midi=108, min_midi=21)

        dac_latents = torch.tensor(dac_latents, dtype=torch.float32)
        return {
            'dac_latents': dac_latents,
            'piano_roll': piano_roll,
            'audio': waveform,  # THIS IS ADDED WITH REFERENCE TO AUDIOLOADER LIBRARY, audio == waveform
            # TODO: diffroll used batch["frame"] which is (piano_roll > 1).float(), based on AudioLoader github
            'frame': (piano_roll > 1).float()
        }


def collate_fn(batch):
    '''pads sequences in the batch to make them same length'''
    dac_latents_list = [item['dac_latents'] for item in batch]
    piano_roll_list = [item['piano_roll'].clone().detach() for item in batch]
    audio_list = [item['audio'] for item in batch]
    frame_list = [item['frame'] for item in batch]
    # Find max lengths
    max_latent_length = max(latent.shape[1] for latent in dac_latents_list)
    max_roll_length = max(roll.shape[0] for roll in piano_roll_list)
    max_audio_length = max(audio.shape[0] for audio in audio_list)
    max_frame_length = max(frame.shape[0] for frame in frame_list)
    # Pad dac_latents and create masks
    padded_dac_latents = []
    dac_masks = []
    for latent in dac_latents_list:
        length = latent.shape[1]
        pad_length = max_latent_length - length
        if pad_length > 0:
            pad = torch.zeros(latent.shape[0], pad_length)
            latent = torch.cat([latent, pad], dim=1)
        padded_dac_latents.append(latent)
        dac_mask = torch.cat([torch.ones(length), torch.zeros(pad_length)])
        dac_masks.append(dac_mask)
    dac_latents = torch.stack(padded_dac_latents)
    dac_masks = torch.stack(dac_masks)

    # Pad piano_roll and create masks
    padded_piano_roll = []
    piano_roll_masks = []
    for roll in piano_roll_list:
        length = roll.shape[0]
        pad_length = max_roll_length - length
        if pad_length > 0:
            pad = torch.zeros(pad_length, roll.shape[1])
            roll = torch.cat([roll, pad], dim=0)
        padded_piano_roll.append(roll)
        piano_roll_mask = torch.cat(
            [torch.ones(length), torch.zeros(pad_length)])
        piano_roll_masks.append(piano_roll_mask)
    piano_roll = torch.stack(padded_piano_roll)
    piano_roll_masks = torch.stack(piano_roll_masks)

    # Pad audio waveforms and create masks
    padded_audio = []
    audio_masks = []
    for audio in audio_list:
        length = audio.shape[0]
        pad_length = max_audio_length - length
        if pad_length > 0:
            pad = torch.zeros(pad_length)
            audio = torch.cat([audio, pad])
        padded_audio.append(audio)
        audio_mask = torch.cat([torch.ones(length), torch.zeros(pad_length)])
        audio_masks.append(audio_mask)
    audio = torch.stack(padded_audio)
    audio_masks = torch.stack(audio_masks)

    # Pad frame and create masks
    padded_frame = []
    frame_masks = []
    for frame in frame_list:
        length = frame.shape[0]
        pad_length = max_frame_length - length
        if pad_length > 0:
            pad = torch.zeros(pad_length)
            frame = torch.cat([frame, pad])
        padded_frame.append(frame)
        frame_mask = torch.cat([torch.ones(length), torch.zeros(pad_length)])
        frame_masks.append(frame_mask)
    frame = torch.stack(padded_frame)
    frame_masks = torch.stack(frame_masks)

    return {
        'dac_latents': dac_latents,
        'piano_roll': piano_roll,
        'dac_masks': dac_masks,
        'piano_roll_masks': piano_roll_masks,
        'audio': audio,
        'audio_masks': audio_masks,
        'frame': frame,
        'frame_masks': frame_masks
    }
