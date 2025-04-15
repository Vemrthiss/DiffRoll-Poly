import gc
import numpy as np
from torch.utils.data import Dataset
from torchaudio.functional import resample
import torchaudio
import pathlib
import torch
import dac
from audiotools import AudioSignal
import librosa

class Custom(Dataset):
    def __init__(
        self,
        audio_path,
        audio_ext,
        max_segment_samples=327680,
        sample_rate=16000,
        with_dac=True
    ):
        r"""Instrument classification dataset takes the meta of an audio
        segment as input, and return the waveform, onset_roll, and targets of
        the audio segment. Dataset is used by DataLoader.
        Args:
            audio_path: str
            audio_ext: str, e.g. mp3, wav, flac
            segment_samples: int, how long you want to cut the audio. If set to None, get the full audio
        """
        audiofolder = pathlib.Path(audio_path)
        self.audio_name_list = list(audiofolder.glob(f'*.{audio_ext}'))

        self.sample_rate = sample_rate
        self.segment_samples = max_segment_samples
        self.with_dac = with_dac
        self.hop_size = 128

        if with_dac:
            model_type = "16khz" if sample_rate == 16000 else "44khz"
            self.dac_model = dac.DAC.load(dac.utils.download(model_type=model_type)).to('cuda')
            CHUNK_DURATION = 60  # seconds
            self.CHUNK_SIZE = sample_rate * CHUNK_DURATION
        

    def __len__(self):
        return len(self.audio_name_list)

    def __getitem__(self, idx):
        r"""Get input and target of a segment for training.
        Args:
            idx for the audio list
        Returns:
          data_dict: {
            'waveform': (samples_num,)
            'onset_roll': (frames_num, classes_num),
            'offset_roll': (frames_num, classes_num),
            'reg_onset_roll': (frames_num, classes_num),
            'reg_offset_roll': (frames_num, classes_num),
            'frame_roll': (frames_num, classes_num),
            'velocity_roll': (frames_num, classes_num),
            'mask_roll':  (frames_num, classes_num),
            'pedal_onset_roll': (frames_num,),
            'pedal_offset_roll': (frames_num,),
            'reg_pedal_onset_roll': (frames_num,),
            'reg_pedal_offset_roll': (frames_num,),
            'pedal_frame_roll': (frames_num,)}
        """
        
        # try:
        waveform, rate = torchaudio.load(self.audio_name_list[idx])
        if waveform.shape[0]==2: # if the audio file is stereo take mean
            waveform = waveform.mean(0) # no need keepdim (audio length), dataloader will generate the B dim
        else:
            waveform = waveform[0] # remove the first dim

        if rate!=self.sample_rate:
            waveform = resample(waveform, rate, self.sample_rate)            
        # except:    
        #     waveform = torch.tensor([[]])
        #     rate = 0
        #     print(f"{self.audio_name_list[idx].name} is corrupted")
            

        data_dict = {}

        # Load segment waveform.
        # with h5py.File(waveform_hdf5_path, 'r') as hf:
        audio_length = len(waveform)
            

        start_sample = 0
        start_time = 0
        end_sample = audio_length
        
        if waveform.shape[0]>=self.segment_samples:
            waveform_seg = waveform[:self.segment_samples]
        else:
            pad = self.segment_samples - waveform.shape[0]
            waveform_seg = torch.nn.functional.pad(waveform, [0,pad], value=0)
        # (segment_samples,), e.g., (160000,)            

        x = torch.randn(1, 640, 88)
        data_dict['waveform'] = waveform_seg
        data_dict['file_name'] = self.audio_name_list[idx].name
        conditioner = waveform_seg
        if self.with_dac:
            librosa_resampled, _ = librosa.load(self.audio_name_list[idx], sr=self.sample_rate)
            # Split audio into 60-second chunks
            num_samples = len(librosa_resampled)
            # num_chunks is number of minutes rounded up (for 60 second chunks)
            num_chunks = int(np.ceil(num_samples / self.CHUNK_SIZE))
            encoded_chunks = []
            signal = AudioSignal(librosa_resampled, sample_rate=self.sample_rate)
            for i in range(num_chunks):
                # Get chunk of audio
                start_idx = i * self.CHUNK_SIZE
                end_idx = min((i + 1) * self.CHUNK_SIZE, num_samples)

                # Create AudioSignal for chunk
                chunk_signal = signal.audio_data[:, :, start_idx:end_idx].to(
                    self.dac_model.device) # 1, 1, T where T = 325591

                # Encode chunk
                s = self.dac_model.preprocess(chunk_signal, signal.sample_rate)
                with torch.no_grad():
                    z, codes, latents, commitment_loss, codebook_loss = self.dac_model.encode(s)

                    encoded_chunks.append((z, codes, latents))

                # Free up GPU memory
                del s, z, codes, latents, commitment_loss, codebook_loss
                gc.collect()
                torch.cuda.empty_cache()
            
            # Concatenate encoded chunks
            conditioner = torch.cat([chunk[2] for chunk in encoded_chunks], dim=-1).squeeze(0) # remove first dim
            del encoded_chunks
            gc.collect()
            torch.cuda.empty_cache()

            # slice latents properly
            begin = 0 # assume start from 0 for simplicity
            step_begin = begin // self.hop_size
            n_steps = self.segment_samples // self.hop_size
            latent_start = step_begin // 4
            num_latent_steps = n_steps // 4
            conditioner = conditioner[:,latent_start:latent_start + num_latent_steps]

            data_dict['dac_latents'] = conditioner
            
        return x, conditioner