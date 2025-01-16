

## Training
- [x] Implement KLD Loss
- [ ] Implement beta-VAE
- [ ] Implement code usage monitor
- [ ] Implement Tau Scheduling
- [ ] Implement Gumbel Hard Scheduling
- [ ] Implement Masking Scheduler
- [ ] Exponential Weighted Iterate Average for parameters (May be already in Pytorch Lightning)

## Checkpointing
- [ ] Save hyperparameters
- [ ] Implement Checkpointing
- [ ] Training Resume

## Disentanglement
- [ ] Implement explicit disentanglement in KLD Loss


## Monitoring
- [ ] Control WandB initialisation and naming
- [ ] Implement Throughput Monitor
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