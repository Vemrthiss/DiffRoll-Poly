#!/bin/bash

python sampling_latent_roll.py \
  dataloader.batch_size=4 \
  dataset=Custom \
  dataset.args.audio_ext=mp3 \
  dataset.args.max_segment_samples=327680 \
  data_root=/teamspace/studios/this_studio/DiffRoll-Poly/docs/transcription