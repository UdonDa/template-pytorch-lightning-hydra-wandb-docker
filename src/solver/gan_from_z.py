import os
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor
from pl_bolts.callbacks import PrintTableMetricsCallback
from src.dataset.data_module import DataModule
from src.models.model import VAE
from torch import Tensor
from torch.optim import Optimizer
from torchvision.utils import save_image

from src.utils.img import denorm

class Solver(pl.LightningModule):
    def __init__(self, config: DictConfig) -> None:
        super(Solver, self).__init__()
        self.config: DictConfig = config

        self.model = VAE(latent_dim=config.latent_dim)
        self.data_module = DataModule(config=config.dataset)

        self.set_network()
        self.validation_z = torch.randn(32, config.generator.nz, 1, 1)

        logger_wandb = instantiate(config.logger)
        logger_wandb.watch(self.G, log = "gradients", log_freq = 100)

        self.trainer = instantiate(
            config.trainer,
            logger=logger_wandb,
            callbacks=[
                LearningRateMonitor(logging_interval="step"),
                instantiate(config.callbacks_image, val_samples=self.validation_z),
                PrintTableMetricsCallback()
            ],
        )

    def set_network(self):
        self.G = instantiate(self.config.generator)
        self.D = instantiate(self.config.discriminator)

    def configure_optimizers(self):
        
        self.optimizer_g = instantiate(
                                self.config.optimizer_g,
                                params=self.G.parameters())
        self.optimizer_d = instantiate(
                                self.config.optimizer_d,
                                params=self.D.parameters())
        
        return [self.optimizer_g, self.optimizer_d], []

    def forward(self, z):
        return self.G(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real = batch

        # sample noise
        z = torch.randn(real.shape[0], self.config.generator.nz, 1, 1)
        z = z.type_as(real)

        # train generator
        if optimizer_idx == 0:

            # generate images
            self.fake = self(z)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(real.size(0), 1, 1, 1)
            valid = valid.type_as(real)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.D(self(z)), valid)
            tqdm_dict = {'g_loss': g_loss}
            output = dict({'loss': g_loss, 'progress_bar': tqdm_dict, 'log': tqdm_dict})

            for k, v in tqdm_dict.items():
                self.log(f"train/G/{k}", v)
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(real.size(0), 1, 1, 1)
            valid = valid.type_as(real)

            real_loss = self.adversarial_loss(self.D(real), valid)

            # how well can it label as fake?
            fake = torch.zeros(real.size(0), 1, 1, 1)
            fake = fake.type_as(real)

            fake_loss = self.adversarial_loss(self.D(self(z).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = dict({'loss': d_loss, 'progress_bar': tqdm_dict, 'log': tqdm_dict})

            for k, v in tqdm_dict.items():
                self.log(f"train/D/{k}", v)
            return output

    def on_train_epoch_end(self):
        return

    def on_epoch_end(self):
        z = self.validation_z.type_as(self.G.main[0].weight)

        # log sampled images
        fake = denorm(self(z)) # [0, 1]
        save_root = os.path.join(self.config['save_root'], 'imgs')
        save_path = os.path.join(save_root, f'{self.current_epoch}.jpg')
        os.makedirs(save_root, exist_ok=True)
        save_image(
            fake,
            save_path,
            nrow=8)
        
    # train your model
    def fit(self):

        self.logger.log_hyperparams(
            {
                "batch_size": self.config.batch_size,
                "lr": self.config.lr,
            }
        )

        self.trainer.fit(self, self.data_module)

    # run your whole experiments
    def run(self):
        self.fit()
