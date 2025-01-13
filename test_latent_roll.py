import hydra
from hydra.utils import to_absolute_path

from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import model as Model

from latentdataset import LatentAudioDataset, collate_fn


@hydra.main(config_path="config", config_name="test_latent_roll")
def main(cfg):
    cfg.data_root = to_absolute_path(cfg.data_root)
    dataset = LatentAudioDataset(
        f'{cfg.data_root}/maestro-latents',
        f'{cfg.data_root}/maestro-v2.0.0'
    )
    _, _, test_set = random_split(dataset, [0.6, 0.2, 0.2])

    test_loader = DataLoader(test_set, collate_fn=collate_fn,batch_size=4)

    # Model
    if cfg.task.frame_threshold != None and cfg.task.sampling.type != None:
        model = getattr(Model, cfg.model.name).load_from_checkpoint(to_absolute_path(
            cfg.checkpoint_path), frame_threshold=cfg.task.frame_threshold, sampling=cfg.task.sampling)
    elif cfg.task.frame_threshold == None and cfg.task.sampling.type != None:
        model = getattr(Model, cfg.model.name).load_from_checkpoint(
            to_absolute_path(cfg.checkpoint_path), sampling=cfg.task.sampling)
    elif cfg.task.frame_threshold != None and cfg.task.sampling.type == None:
        model = getattr(Model, cfg.model.name).load_from_checkpoint(
            to_absolute_path(cfg.checkpoint_path), frame_threshold=cfg.task.frame_threshold)
    else:
        model = getattr(Model, cfg.model.name).load_from_checkpoint(
            to_absolute_path(cfg.checkpoint_path))

    if cfg.model.name == 'ClassifierFreeDiffRoll':
        name = f"Test-x0_pred_0-{cfg.model.name}-" \
               f"{cfg.task.sampling.type}-w{cfg.task.sampling.w}-{cfg.dataset.name}"
        logger = TensorBoardLogger(save_dir=".", version=1, name=name)
    else:
        name = f"Test-x0_pred_0-{cfg.model.name}-" \
               f"{cfg.task.sampling.type}-{cfg.dataset.name}"
        logger = TensorBoardLogger(save_dir=".", version=1, name=name)

    trainer = pl.Trainer(logger=logger)

    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
