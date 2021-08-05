from pytorch_lightning.utilities import rank_zero_info
from torchvision.utils import make_grid

import pytorch_lightning as pl
import wandb
import torch
import copy


class WandbImageFromZCallback(pl.Callback):
    """Logs the input and output images of a module.
    
    Images are stacked into a mosaic, with output on the top
    and input on the bottom."""
    
    def __init__(self, val_samples, max_samples=32):
        super().__init__()
        self.val_imgs = val_samples
        self.val_imgs = self.val_imgs[:max_samples]
    
    @torch.no_grad()
    def on_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)
    
        pl_module.eval()
        outs = pl_module(val_imgs)
        pl_module.train()

        grid = make_grid(outs, nrow=8, normalize=True)
    
        caption = "Generated Images from randn"
        trainer.logger.experiment.log({
            "val/examples": wandb.Image(grid, caption=caption),
            "global_step": trainer.global_step
            })