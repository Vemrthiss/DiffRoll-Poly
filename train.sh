#!/bin/bash

python train_latent_roll.py "+gpus=[0]" model.args.kernel_size=9 dataset=MAESTRO dataloader.train.num_workers=4 epochs=2500 download=False