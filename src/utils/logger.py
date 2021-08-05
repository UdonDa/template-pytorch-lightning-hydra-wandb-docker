
class ImageSegmentationLogger(Callback):
    def __init__(self, val_samples, num_samples=8,log_interval=5):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples
        self.log_interval = log_interval

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)

        #[B,C,Z,Y,X]
        pred_probs = pl_module(val_imgs)
        #[B,Z,Y,X] -> [B,Y,X]
        preds = torch.argmax(pred_probs, 1)[:,0,...].cpu().numpy()
        val_labels =torch.argmax(val_labels, 1) [:,0,...].cpu().numpy()

        class_labels = {
            0: "gd",
            1: "kidney",
            2: "cancer",
            3: "cyst"
        }

        # Log the images as wandb Image
        trainer.logger.experiment.log({
            "examples": [wandb.Image(x, masks={
                "predictions": {
                    "mask_data": pred,
                    "class_labels": class_labels
                },
                "groud_truth": {
                    "mask_data": y,
                    "class_labels": class_labels
                }
            })
                for x, pred, y in zip(val_imgs[:self.num_samples],
                                    preds[:self.num_samples],
                                    val_labels[:self.num_samples])]
        })