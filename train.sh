#!/bin/bash

python train_latent_roll.py \
    model.args.kernel_size=9 \
    dataloader.train.num_workers=4 \
    epochs=3500 \
    download=False \
    trainer.devices=4 \
    trainer.strategy=ddp \
    trainer.log_every_n_steps=10

# python train_spec_roll.py \
#     +gpus=[0] \
#     model.args.kernel_size=9 \
#     model.args.spec_dropout=0.1 \
#     dataloader.train.num_workers=4 \
#     epochs=2500 \
#     download=False
