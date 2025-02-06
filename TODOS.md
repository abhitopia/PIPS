

## Debug 
- [ ] Why is TC loss negative?

## Training
- [x] Implement KLD Loss
- [x] Implement beta-TCVAE
- [x] Implement ReinMax as an alternative to Gumbel Hard (Algorithm 2)
- [ ] Codebook Weight Initialisation
- [x] Implement Tau Scheduling
- [ ] Implement code usage monitor
- [ ] Implement Gumbel Hard Scheduling
- [ ] Implement Masking Scheduler
- [ ] Exponential Weighted Iterate Average for parameters (May be already in Pytorch Lightning)


## Checkpointing
- [ ] Save hyperparameters
- [ ] Implement Checkpointing
- [ ] Prevent saving everything or at least deleting unnecessary artifacts
- [ ] Training Resume

## Disentanglement
- [x] Implement explicit disentanglement in KLD Loss


## Monitoring
- [ ] Control WandB initialisation and naming
- [x] Implement Throughput Monitor
- [ ] Monitoring number of different codes in pair of input and output grids from eval set
- [ ] Implement disentanglement monitor

## CLI
- [ ] Add Seeding Everything
- [ ] Check the Dalle-E Paper for default parameters and schedules
- [ ] Check Dalle-E paper for how they handle loss component scaling

## Refactor
- [ ] Remove pesky warnings

## Optimisation
- [ ] Optimise training for speed

## Grokking
- [ ] Implement Orthograd Optimizer
- [ ] Use 32bit float CE loss

## Archived
- [ ] Dynamically adjust the beta? (It should be possible to do this by monitoring the rate of decay of the two loss components and adjusting the beta accordingly)