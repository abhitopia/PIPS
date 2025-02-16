

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
- [x] PAD/EOS value to be determined automatically from Model Config
- [x] Move ProjectSize to Model Config
- [x] Make gradient clipping specified in the experiment config
- [x] Gradient Accumulation and how to do it in Lightning
- [x] Make optimiser fused dependent on cuda availability
- [x] Make the dataloader work with workrs > 0 faster
- [x] Update CLI defaults
- [x] Update ExperimentConfig defaults
- [x] Check the Dalle-E Paper for default parameters and schedules
    - [x] AdamW b1 0.9, b2 0.999, eps 1e-8
    - [x] KL Beta 0 to 6.6 Warmup 5000 steps (I use 10_000)
    - [x] Learning Rate 1e-4 -> 1.125e-6  over 1.2M steps (I will set the min LR to 0.01, approx /100 of the max LR)
    - [x] Temperature from 1.0 - > 1/16 (0.0625) 
    - [x] Temperature warmup over 150_000 steps. Cosine anneal
    - [x] Total 3M steps, per GPU batch size 8, multi-gpu batchsize 512 (I will use 1M steps)
    - [x] Batch Size 512 (I will use 8 GPUs so larger batch size is possible)
- [x] Fix the losses, remove them from betas
- [x] Scale the KLD loss by the one over number of latents
- [x] Make relu to kld losses optional parameter
- [ ] Regularise only 2D parameters with weight decay (skip biases and BatchNorm or LayerNorm)


## Checkpointing
- [x] Implement Checkpointing
- [x] Prevent saving everything or at least deleting unnecessary artifacts (Write a Wandb Sync Callback)
- [x] Add rudimentary resume support from local checkpoint
- [x] Save hyperparameters
- [x] Training Resume from WandB
- [x] Add support for new from checkpoint CLI command
- [x] Refactor Wandb Checkpoint Sync to use Artifact class
- [x] Refactor new to use model from project/run_name/model-name/alias|step
- [x] Change ExperimentConfig to save project/run_name/model-name/alias|step
- [x] Make resume command use the same logic as new command for specifying remote source. That way there is less code duplication

## Disentanglement
- [x] Implement explicit disentanglement in KLD Loss
- [x] Relu out the negative losses (probably due to approximation)


## Monitoring
- [x] Monitor masking percentage
- [x] Implement Throughput Monitor
- [x] Control WandB initialisation and naming
- [x] Add debug mode to disable logging to WANDB (may be ligthning internal)
- [x] Monitor learning rate
- [x] Implement Token Accuracy and Sample Accuracy 
- [x] Non EOS token accuracy as well for validation
- [x] Rename Accuracy logs
- [x] Log learning rate (Already done)
- [ ] Monitoring number of different codes in pair of input and output grids from eval set
- [ ] Implement disentanglement monitor
- [ ] Implement input/output token accuracy
- [ ] Implement code usage monitor


## CLI
- [x] Add new training CLI command
- [x] Add resume CLI command
- [x] Add Seeding Everything
- [x] Make cli.py a cli app
- [x] Add lr finder CLI command
- [x] Refactor CLI to use common options
- [x] Refactor CLI to specify Acceleration Config
- [x] Rename parameters

## Refactor
- [ ] Remove pesky warnings

## Optimisation
- [x] Set up pytorch lightning studio
- [x] https://lightning.ai/docs/pytorch/stable/accelerators/gpu.html
    - L40S 
        - BS: 32 -> 174k
- [x] `torch.set_float32_matmul_precision('high')` for better performance.
    - L40S: BS: 32 -> 200k
- [x] Move the q_z_marg to be outside of the module.
- [x] Generate multiple inputs for test_compile.py and set requires_grad to False for the inputs.
- [x] [torch.compile](https://lightning.ai/docs/pytorch/stable/advanced/compile.html#apply-torch-compile-to-your-lightningmodule)
    - L40: BS: 32 -> 1.5M (compile) and 56k (no compile)
- [x] N-bit precision https://lightning.ai/docs/pytorch/stable/common/precision.html
- [x] https://lightning.ai/docs/pytorch/stable/accelerators/accelerator_prepare.html
- [x] Optimise training for speed
- [x] Multi-GPU training
- [x] Add dist log sync for validation
- [ ] Use multiple threads on dataloader using sharing datasets across process boundaries
- [ ] https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#sharing-datasets-across-process-boundaries
- [ ] Profile https://lightning.ai/docs/pytorch/stable/tuning/profiler.html

## Grokking
- [ ] Implement Orthograd Optimizer
- [ ] Use 32bit float CE loss

## Debug/Issues
- [ ] Why is TC loss negative?
    - Come back to it when training. Probably due to approximation, large batch size may fix it.

## Archived
- [ ] Dynamically adjust the beta? (It should be possible to do this by monitoring the rate of decay of the two loss components and adjusting the beta accordingly)
- [ ] Positional Embedding may be needed on top of the encoder output because all the latents come from the same codebook (Should not be needed as queries in pooler should double up as positional encodings)
- [ ] Codebook regularization (to bound it) (This can be done by adding weight decay to start with)
- [ ] Codebook Weight Initialisation
- [ ] Add evaluate CLI command


## Later
- [ ] Exponential Weighted Iterate Average for parameters (May be already in Pytorch Lightning) (Will use SWA instead)
- [ ] https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#stochastic-weight-averaging