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
from torch.optim.lr_scheduler import MultiStepLR

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


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
            self.mlp = MLP(n_in=n_in, act=act,
                           act_trainable=hparams.act_trainable,
                           **kwargs)

        elif hparams.arch == 'ff':
            P = hparams.sc*torch.normal(torch.zeros(2, 256),
                                        torch.ones(2, 256)) # (2, 256)
            self.pe = PE(P)
            self.mlp = MLP(n_in=self.pe.out_dim)

        elif hparams.arch == 'siren':
            self.mlp = Siren(first_omega_0=hparams.omega_0,
                             hidden_omega_0=hparams.omega_0)

        self.loss = nn.MSELoss()
        
    def forward(self, x):
        if hparams.use_pe or hparams.arch=='ff':
            x = self.pe(x)
        return self.mlp(x)
        
    def setup(self, stage=None):
        self.train_dataset = ImageDataset(hparams.image_path,
                                          hparams.img_wh,
                                          'train')
        self.val_dataset = ImageDataset(hparams.image_path,
                                        hparams.img_wh,
                                        'val')

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
        act_layers = ['net.1.a', 'net.3.a', 'net.5.a',
                      'net.1.b', 'net.3.b', 'net.5.b'] # hard-coded
        act_params = map(lambda x: x[1],
                         filter(lambda kv: kv[0] in act_layers,
                                self.mlp.named_parameters()))
        net_params = map(lambda x: x[1],
                         filter(lambda kv: kv[0] not in act_layers,
                                self.mlp.named_parameters()))

        self.net_optimizer = Adam(net_params, lr=self.hparams.lr)
        self.act_optimizer = Adam(act_params, lr=self.hparams.lr_a)

        if hparams.use_lr_a_decay:
            act_scheduler = MultiStepLR(self.act_optimizer,
                                        [hparams.num_epochs//2,
                                         hparams.num_epochs*3//4],
                                        gamma=0.1)

            return [self.net_optimizer, self.act_optimizer], [act_scheduler]

        return [self.net_optimizer, self.act_optimizer]

    def training_step(self, batch, batch_idx, optimizer_idx):
        rgb_pred = self(batch['uv'])

        loss = self.loss(rgb_pred, batch['rgb'])
        psnr_ = psnr(rgb_pred, batch['rgb'])

        self.log('lr/net', get_learning_rate(self.net_optimizer))
        self.log('lr/act', get_learning_rate(self.act_optimizer))
        self.log('train/loss', loss)
        self.log('train/psnr', psnr_, prog_bar=True)

        if hparams.arch in ['gaussian', 'quadratic',
                            'multi-quadratic', 'laplacian',
                            'super-gaussian', 'expsin']:
            for i in [1, 3, 5]: # activation layers
                self.log(f'act/l{i}_a', self.mlp.net[i].a)
                if hasattr(self.mlp.net[i], "b"):
                    self.log(f'act/l{i}_b', self.mlp.net[i].b)

        return loss

    def validation_step(self, batch, batch_idx):
        rgb_pred = self(batch['uv'])

        loss = self.loss(rgb_pred, batch['rgb'])
        psnr_ = psnr(rgb_pred, batch['rgb'])

        log = {'val_loss': loss,
               'val_psnr': psnr_,
               'rgb_gt': batch['rgb'],
               'rgb_pred': rgb_pred}

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        rgb_gt = torch.cat([x['rgb_gt'] for x in outputs])
        rgb_gt = rearrange(rgb_gt, '(h w) c -> c h w',
                           h=hparams.img_wh[1],
                           w=hparams.img_wh[0])
        rgb_pred = torch.cat([x['rgb_pred'] for x in outputs])
        rgb_pred = rearrange(rgb_pred, '(h w) c -> c h w',
                             h=hparams.img_wh[1],
                             w=hparams.img_wh[0])

        self.logger.experiment.add_images('val/gt_pred',
                                          torch.stack([rgb_gt, rgb_pred]),
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