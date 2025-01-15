import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from pips.grid_dataset import GridDataset
from pips.dvae import GridDVAEConfig, GridDVAE
import torch
from functools import partial
import torch.nn.functional as F


class DVAETrainingModule(pl.LightningModule):
    def __init__(self, config):
        super(DVAETrainingModule, self).__init__()
        self.config = config
        self.model = self.build_model()

    def build_model(self):
        return GridDVAE(self.config)

    def forward(self, x):
        mask_percentage = 0.0
        hard = True
        tau = 0.9
        # Create a random boolean mask
        attn_mask = self.model.create_random_mask(x.size(0), x.size(1), mask_percentage, same_mask_for_all=True)
        code, encoded_logits = self.model.encode(x, attn_mask, tau, hard)

        # Encoded Logits have the shape (B, N, C)
        # B is the batch size
        # N is the number of discrete codes per sample
        # C is the codebook size


        logP = F.log_softmax(encoded_logits, dim=-1)                                 # (B, N, C)
        logU = torch.log(torch.ones_like(logP) / self.config.codebook_size)          # (B, N, C)


        # We use KLDivLoss in log_target=True mode:
        #   loss(input=logQ, target=logP) = sum_i( P_i * [logP_i - logQ_i] ) => D(P||Q).
        #   Here, Q=U, so input=log_u; P is from encoder, so target=log_p.
        elbo_loss_per_sample = torch.nn.functional.kl_div(
            input=logU,           # = logU (Q)
            target=logP,          # = logP (P)  
            log_target=True,     
            reduction="batchmean"
        )

        print("ELBO Loss per sample:", elbo_loss_per_sample)

        decoded_logits = self.model.decode(code) # (B, S, V)

        # Reconstruction Loss
        ce_sum_per_sample = F.cross_entropy(
            decoded_logits.view(-1, decoded_logits.size(-1)),
            x.view(-1),
            reduction='sum'
        )/ x.size(0) # Divide by batch size to get the average loss per sample

        print("Reconstruction Loss per sample:", ce_sum_per_sample)

        total_loss = elbo_loss_per_sample + ce_sum_per_sample  
        print("Total Loss:", total_loss)
        return decoded_logits, total_loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, loss = self(x)
        self.log('train_loss', loss, batch_size=x.size(0), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, loss = self(x)
        self.log('val_loss', loss, batch_size=x.size(0))

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

def main():
    batch_size = 4
    pad_value = 10
    project_size = (32, 32)
    eval_every_steps = 1000
    # Define the configuration for the DVAE
    config = GridDVAEConfig(
        n_dim=256,
        n_head=8,
        n_layers=6,
        n_codes=16,
        codebook_size=512,
        rope_base=10000,
        dropout=0.0,
        n_pos=project_size[0] * project_size[1],
        n_vocab=16
    )

    # Create the dataset and dataloader
    collate_fn_train = partial(GridDataset.collate_fn, pad_value=pad_value, permute=True, project_size=project_size)
    train_dataset = GridDataset(train=True)
    train_loader = DataLoader(train_dataset, 
                              collate_fn=collate_fn_train,
                              batch_size=batch_size, 
                              shuffle=True, num_workers=0)

    collate_fn_val = partial(GridDataset.collate_fn, pad_value=pad_value, permute=False, project_size=project_size)
    val_dataset = GridDataset(train=False)
    val_loader = DataLoader(val_dataset, 
                            collate_fn=collate_fn_val,
                            batch_size=batch_size,
                            persistent_workers=True,
                            shuffle=False, 
                            num_workers=4)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Initialize the Lightning module
    model = DVAETrainingModule(config)

    # Set up Weights and Biases logger
    wandb_logger = WandbLogger(project='dvae-training', log_model=True)

    # Initialize the trainer with fast_dev_run for debugging
    trainer = pl.Trainer(
        fast_dev_run=False,  # Run a sin    gle batch for training, validation, and testing
        accelerator="gpu" if torch.cuda.is_available() else "cpu",  # Use CUDA if available, otherwise CPU
        devices=1,  # Use one GPU or CPU
        logger=wandb_logger,
        gradient_clip_val=1.0,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor='train_loss')],
        # val_check_interval=eval_every_steps  # Run validation every 1000 training steps
        # max_steps=10000,    # Limit the number of training steps

        max_steps=20,
        limit_train_batches=10,
        limit_val_batches=10,
        val_check_interval=5,
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)  # Ensure val_loader is passed if validation is needed

if __name__ == '__main__':
    main() 