

## Training
- [x] Implement KLD Loss
- [x] Implement beta-TCVAE
- [x] Implement ReinMax as an alternative to Gumbel Hard (Algorithm 2)
- [x] Implement Tau Scheduling
- [x] Implement Gumbel Hard Scheduling
- [x] Implement Masking Scheduler
- [x] Add KL to the Loss along with the schedule
- [x] Switched to using AdamW with weight decay
- [x] Add Learning Rate Noam Schedule
- [ ] PAD/EOS value to be determined automatically from Model Config
- [ ] Codebook Weight Initialisation
- [ ] Codebook regularization (to bound it)
- [ ] Implement code usage monitor
- [ ] Exponential Weighted Iterate Average for parameters (May be already in Pytorch Lightning)


## Checkpointing
- [x] Implement Checkpointing
- [x] Prevent saving everything or at least deleting unnecessary artifacts (Write a Wandb Sync Callback)
- [ ] Save hyperparameters
- [ ] Training Resume

## Disentanglement
- [x] Implement explicit disentanglement in KLD Loss
- [x] Relu out the negative losses (probably due to approximation)


## Monitoring
- [x] Monitor masking percentage
- [x] Implement Throughput Monitor
- [x] Control WandB initialisation and naming
- [x] Add debug mode to disable logging to WANDB (may be ligthning internal)
- [x] Monitor learning rate
- [ ] Implement disentanglement monitor
- [ ] Non EOS token accuracy as well for validation
- [ ] Monitoring number of different codes in pair of input and output grids from eval set

## CLI
- [ ] Add Seeding Everything
- [ ] Check the Dalle-E Paper for default parameters and schedules
- [ ] Check Dalle-E paper for how they handle loss component scaling

## Refactor
- [x] Remove pesky warnings

## Optimisation
- [ ] Optimise training for speed

## Grokking
- [ ] Implement Orthograd Optimizer
- [ ] Use 32bit float CE loss

## Debug/Issues
- [ ] Why is TC loss negative?
    - Come back to it when training. Probably due to approximation, large batch size may fix it.

## Archived
- [ ] Dynamically adjust the beta? (It should be possible to do this by monitoring the rate of decay of the two loss components and adjusting the beta accordingly)
- [ ] Positional Embedding may be needed on top of the encoder output because all the latents come from the same codebook (Should not be needed as queries in pooler should double up as positional encodings)