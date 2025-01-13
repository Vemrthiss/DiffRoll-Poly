#!/bin/bash

python sampling_latent_roll.py \
  task=transcription \
  dataloader.batch_size=4 \
  dataset=Custom \
  dataset.args.audio_ext=mp3 \
  dataset.args.max_segment_samples=327680 \
  data_root=./my_audio