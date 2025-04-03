import hydra
from hydra.utils import to_absolute_path

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import model as Model

import AudioLoader.music.amt as MusicDataset
from AudioLoader.music.amt import ChunkedDataset

num_chunks=4

@hydra.main(config_path="config", config_name="test_latent_roll")
def main(cfg):
    cfg.data_root = to_absolute_path(cfg.data_root)
    cfg.latent_dir = to_absolute_path(cfg.latent_dir)

    # test_set = getattr(MusicDataset, cfg.dataset.name)(**cfg.dataset.test)
    # test_loader = DataLoader(test_set, batch_size=4)
    test_set = ChunkedDataset((getattr(MusicDataset, cfg.dataset.name)(**cfg.dataset.test)), num_chunks=num_chunks)
    test_loader = DataLoader(test_set, batch_size=4) # was batch size 4

    # Model
    if cfg.task.frame_threshold != None and cfg.task.sampling.type != None:
        model = getattr(Model, cfg.model.name).load_from_checkpoint(to_absolute_path(
            cfg.checkpoint_path), frame_threshold=cfg.task.frame_threshold, sampling=cfg.task.sampling, timesteps=cfg.task.timesteps)
    elif cfg.task.frame_threshold == None and cfg.task.sampling.type != None:
        model = getattr(Model, cfg.model.name).load_from_checkpoint(
            to_absolute_path(cfg.checkpoint_path), sampling=cfg.task.sampling)
    elif cfg.task.frame_threshold != None and cfg.task.sampling.type == None:
        model = getattr(Model, cfg.model.name).load_from_checkpoint(
            to_absolute_path(cfg.checkpoint_path), frame_threshold=cfg.task.frame_threshold)
    else:
        model = getattr(Model, cfg.model.name).load_from_checkpoint(
            to_absolute_path(cfg.checkpoint_path))

    if cfg.model.name == 'ClassifierFreeLatentRoll':
        name = f"Test-x0_pred_0-{cfg.model.name}-" \
               f"{cfg.task.sampling.type}-w{cfg.task.sampling.w}-{cfg.dataset.name}"
    else:
        name = f"Test-x0_pred_0-{cfg.model.name}-" \
               f"{cfg.task.sampling.type}-{cfg.dataset.name}"

    logger = TensorBoardLogger(save_dir=".", version=1, name=name)
    trainer = pl.Trainer(
            logger=logger,
            devices=4,
            strategy='ddp',
            log_every_n_steps=10
        )

    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
