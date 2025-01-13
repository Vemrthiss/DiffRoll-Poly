import hydra
from hydra.utils import to_absolute_path
import model as Model
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from latentdataset import LatentAudioDataset, collate_fn


@hydra.main(config_path="config", config_name="latent_roll")
def main(cfg):
    cfg.data_root = to_absolute_path(cfg.data_root)

    dataset = LatentAudioDataset(
        f'{cfg.data_root}/maestro-latents',
        f'{cfg.data_root}/maestro-v2.0.0'
    )
    train_set, val_set, test_set = random_split(dataset, [0.6, 0.2, 0.2])
    # print('SANTIY CHECK')
    # TODO: recall there might be a problem with aligning the latents and piano rolls?
    # dac_latents: torch.Tensor = train_set[1]['dac_latents']
    # piano_roll: torch.Tensor = train_set[1]['piano_roll']
    # print(torch.count_nonzero(piano_roll))

    train_loader = DataLoader(
        train_set, collate_fn=collate_fn, **cfg.dataloader.train)
    val_loader = DataLoader(
        val_set, collate_fn=collate_fn, **cfg.dataloader.val)
    test_loader = DataLoader(
        test_set, collate_fn=collate_fn, **cfg.dataloader.test)

    # Model
    # TODO: no latent_args for now
    # model = getattr(Model, cfg.model.name)(**cfg.model.args,
    #                                        latent_args=cfg.latent.args, **cfg.task)
    model = getattr(Model, cfg.model.name)(**cfg.model.args, **cfg.task)

    optimizer = Adam(model.parameters(), lr=1e-3)

    checkpoint_callback = ModelCheckpoint(**cfg.modelcheckpoint)

    if cfg.model.name == 'DiffRollBaseline':
        name = f"{cfg.model.name}-L{cfg.model.args.residual_layers}-C{cfg.model.args.residual_channels}-" + \
               f"t={cfg.task.time_mode}-x_t={cfg.task.x_t}-k={cfg.model.args.kernel_size}-" + \
               f"dia={cfg.model.args.dilation_base}-{cfg.model.args.dilation_bound}-" + \
               f"{cfg.dataset.name}"
    elif cfg.model.name == 'ClassifierFreeDiffRoll':
        name = f"{cfg.model.name}-L{cfg.model.args.residual_layers}-C{cfg.model.args.residual_channels}-" + \
               f"beta{cfg.task.beta_end}-{cfg.task.training.mode}-" + \
               f"{cfg.task.sampling.type}-w={cfg.task.sampling.w}-" + \
               f"p={cfg.model.args.spec_dropout}-k={cfg.model.args.kernel_size}-" + \
               f"dia={cfg.model.args.dilation_base}-{cfg.model.args.dilation_bound}-" + \
               f"{cfg.dataset.name}"
    else:
        name = f"{cfg.model.name}-{cfg.task.sampling.type}-L{cfg.model.args.residual_layers}-C{cfg.model.args.residual_channels}-" + \
               f"beta{cfg.task.beta_end}-{cfg.task.training.mode}-" + \
               f"dilation{cfg.model.args.dilation_base}-{cfg.task.loss_type}-{cfg.dataset.name}"
    logger = TensorBoardLogger(save_dir=".", version=1, name=name)

    trainer = pl.Trainer(**cfg.trainer,
                         callbacks=[checkpoint_callback,],
                         logger=logger)

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
