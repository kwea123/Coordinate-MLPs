import torch
from torch import nn
from einops import rearrange

from opt import get_opts

# datasets
from dataset import ImageDataset
from torch.utils.data import DataLoader

# models
from models import PE, MLP, Siren

# metrics
from metrics import psnr

# optimizer
from torch.optim import Adam

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger


class CoordMLPSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        if hparams.use_pe:
            P = torch.cat([torch.eye(2)*2**i for i in range(10)], 1) # (2, 2*10)
            self.pe = PE(P)

        if hparams.arch in ['relu', 'gaussian', 'quadratic',
                            'multi-quadratic', 'laplacian',
                            'super-gaussian', 'expsin']:
            kwargs = {'a': hparams.a, 'b': hparams.b}
            act = hparams.arch
            if hparams.use_pe:
                n_in = self.pe.out_dim
            else:
                n_in = 2
            self.net = MLP(n_in=n_in, act=act, **kwargs)

        elif hparams.arch == 'ff':
            P = hparams.sc*torch.normal(torch.zeros(2, 256),
                                        torch.ones(2, 256)) # (2, 256)
            self.pe = PE(P)
            self.net = MLP(n_in=self.pe.out_dim)

        elif hparams.arch == 'siren':
            self.net = Siren(first_omega_0=hparams.omega_0,
                             hidden_omega_0=hparams.omega_0)

        self.loss = nn.MSELoss()
        
    def forward(self, x):
        if hparams.use_pe or hparams.arch=='ff':
            x = self.pe(x)
        return self.net(x)
        
    def setup(self, stage=None):
        self.train_dataset = ImageDataset(hparams.image_path, 'train')
        self.val_dataset = ImageDataset(hparams.image_path, 'val')

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def configure_optimizers(self):
        self.optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)

        return self.optimizer

    def training_step(self, batch, batch_idx):
        rgb_pred = self(batch['uv'])

        loss = self.loss(rgb_pred, batch['rgb'])
        psnr_ = psnr(rgb_pred, batch['rgb'])

        self.log('train/loss', loss)
        self.log('train/psnr', psnr_, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        rgb_pred = self(batch['uv'])

        loss = self.loss(rgb_pred, batch['rgb'])
        psnr_ = psnr(rgb_pred, batch['rgb'])

        log = {'val_loss': loss,
               'val_psnr': psnr_,
               'rgb_pred': rgb_pred} # (B, 3)

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        rgb_pred = torch.cat([x['rgb_pred'] for x in outputs]) # (512*512, 3)
        rgb_pred = rearrange(rgb_pred, '(h w) c -> c h w',
                             h=self.train_dataset.r,
                             w=self.train_dataset.r)

        self.logger.experiment.add_image('val/image_pred',
                                         rgb_pred,
                                         self.global_step)

        self.log('val/loss', mean_loss, prog_bar=True)
        self.log('val/psnr', mean_psnr, prog_bar=True)


if __name__ == '__main__':
    hparams = get_opts()
    system = CoordMLPSystem(hparams)

    # ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}',
    #                           filename='{epoch:d}',
    #                           save_top_k=-1)
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [pbar]

    logger = TensorBoardLogger(save_dir="logs",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=True,
                      accelerator='auto',
                      devices=1,
                      num_sanity_val_steps=0,
                      log_every_n_steps=1,
                      check_val_every_n_epoch=20,
                      benchmark=True)

    trainer.fit(system)