import hydra
from hydra.utils import to_absolute_path
import model as Model
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import AudioLoader.music.amt as MusicDataset


@hydra.main(config_path="config", config_name="latent_roll")
def main(cfg):
    cfg.data_root = to_absolute_path(cfg.data_root)
    cfg.latent_dir = to_absolute_path(cfg.latent_dir)

    train_set = getattr(MusicDataset, cfg.dataset.name)(**cfg.dataset.train)
    val_set = getattr(MusicDataset, cfg.dataset.name)(**cfg.dataset.val)
    test_set = getattr(MusicDataset, cfg.dataset.name)(**cfg.dataset.test)

    train_loader = DataLoader(train_set, **cfg.dataloader.train)
    val_loader = DataLoader(val_set, **cfg.dataloader.val)
    test_loader = DataLoader(test_set, **cfg.dataloader.test)

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

    trainer.fit(
        model,
        train_loader,
        val_loader,
        ckpt_path="/teamspace/studios/this_studio/DiffRoll-Poly/outputs/2025-03-03/16-03-20/ClassifierFreeLatentRoll-cfdg_ddpm_x0-L15-C512-beta0.02-x_0-dilation2-l2-MAESTRO/version_1/checkpoints/last.ckpt"
    )
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
