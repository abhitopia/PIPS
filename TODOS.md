# 31st March 2025 (Milestone 1)
- VQ VAE with EMA and codebook resets seems to be working finally!!!!!!!!
- [x] Add distance reset to the codebook

# EMA Codeboook
- [x] Add codebook resets
- [x] Measure peplexity/entropy
- [ ] log the norm of the encoder vs codebook

# 28th March 2025 (Add VQVAE)
- [x] Add VQVAE and corresponding CLI

# 27th March 2025
- [x] Gumbel Scaling should be zero for validation
- [x] Implement EMA for the codebook
- [ ] Consider changing the rope base to lower value in future

# 23rd March 2025
- [x] Implement 4 different loss functions (divergence sample, batch, entropy, position)
- [x] Implement ability to run CLI using configs



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
- [x] Regularise only 2D parameters with weight decay (skip biases and BatchNorm or LayerNorm). Change default weight decay to 1e-4
- [x] Add the ability to override batch size and learning rate
- [x] Add the val_check_interval to the CLI


## Debug/Issues
- [x] Add normalisations to the pooling layers
- [x] Change the initialisation of the pooling layer queries to match something else? 
- [x] Log the model gradients
- [x] Add residual connection to the pooling layers
- [x] Create ResidualProjection class 
- [x] Add tests for the ResidualProjection class
- [x] Remove AttentionPool, add TransformerProjection
- [x] Add tests for the TransformerProjection class
- [x] Change the mask shape to not be attention mask for ResidualProjection
- [x] Change the mask shape to not be attention mask for TransformerProjection
- [x] Modify stacked pooling to StackedTransformerProjection be simpler and use TransformerProjection
- [x] Adjust tests for the new stacked pooling
- [x] Fix transformer tests
- [x] Upgrade rest of the code to use StackedTransformerProjection
- [x] Fix all the remaining tests
- [x] Remove feature normalisation from the residual projection
- [x] Check if token normalisation done the way it is makes sense. Changed to Mask Normalisation
- [x] Make the output normalisation from transformer optional
- [x] Make the output normalisation from StackedTransformerProjection optional
- [x] Correctly handle output normalisation in the DVAE
- [x] Make mask normalisation a parameter in Config
- [x] Update the flatten function to Grid to handle padding and EOS
- [x] Change the collate function to use the new flatten function
- [x] Introduce eos_value in the config
- [x] Change the loss to discount pad_value
- [x] introduce soft to hard interpolation
- [x] Test with vanilla softmax!
- [x] Add Karpathy's GPU initialisation for the network
- [x] Change starting tau initialisation to 3.5
- [x] Introduce disable permute flag in the CLI
- [x] Fix cosine anneal and cosine decay to be just one function
- [x] 2D positions incorrect after the changed dataset
- [ ] Determine the cause for code collapse
- [ ] Profile the code, the number of iterations per second seems to vary a lot
- [ ] Why is TC loss negative?
    - Come back to it when training. Probably due to approximation, large batch size may fix it.


## Version 2
- [x] Add Learning rate final
- [x] Move TC RELU application to the TrainDVAE class
- [x] Add padding token CE loss weighting
- [x] Add masking mechanism 
- [x] Fix all the tests
- [x] Clean up the code
- [x] Move entropy calculation to the TrainDVAE class
- [x] Perplexity per code
- [x] Histogram of code usage per code
- [x] Log reconstruction images to the validation set
- [x] Make cude synchronisation optional and configurable
- [x] Add use_exp_relaxed flag and use_monte_carlo_kld flag to CLI
- [x] Log loss as a whole as well
- [x] Add ability to scale the CE loss
- [ ] Verify data
- [ ] Verify masked data and reconstruction loss
- [ ] Adding padding token accuracy
- [ ] Add padding xeight schedule


## Version 3 (Make the logalpha peaky)
- [x] Visualise with and without tau distribution
- [x] Add entropy penalty and corresponding coefficient and schedulers
- Model doesn't train even without any entropy penalty or temperature.
- [x] Add Option to skip codebook altogether
- [x] bias=True for value project in codebook
- [x] Add positional encoding to the codebook
- [x] Normalise the keys and queries so network doesn't fight back the decrease in temperature
- [ ] Create a MNIST example (properly this time)


## Version 4
- [x] Add temperature to the entropy loss
- [x] Fix all perplexity and entropy calculations to only use log_alpha / tau
- [x] Implement diversity loss
- [x] Plot peakiness over steps
- [x] Fix perplexity to use base 2
- [x] Add global step number to the images in the wandb logs (log tau and noise scale)
- [x] Add gumbel noise option to code, which changes the log_alpha before the softmax
- [x] Change all the functions to work independently of the tau
- [x] simplify the code

# Fix KLD Losses
- [x] Read the paper
- [x] Use log_softmax for numerical stability
- [x] Use distributions if you have to
- [x] KLD loss uses non-temperature scaled logits (verify this)
- [x] Nan on approximate KLD loss
- [x] Perplexity is Inf
- [x] Add ability to limit the number of samples in the dataset
- [x] Change the cli to use the new max_training_samples parameter instead of limit_train_batches
- [x] Allow configurable permutation
- [x] Measure avg grid information content
- [x] Add focal loss
- [x] Remove power of 2 restriction on the number of latent codes   
- [x] Different Rope base for different axes
- [x] Sample accuracy includes padding tokens
- [x] Add soft codebook (ignores temperature altogether)
- [x] Add sampling flag to the codebook, temperature is applied regardless of sampling or not. For regular softmax, keep temperature
at 1.0
- [x] Make forward pass of DVAEModule only take tensors and no python variables, and output only tensors
- [x] fix following
```
    skipping cudagraphs due to skipping cudagraphs due to cpu device (primals_88). Found from : 
   File "/teamspace/studios/this_studio/PIPS/pips/dvae.py", line 782, in forward
    quantized, log_alpha, _ = self.codebook(encoded_logits, tau=tau)
  File "/teamspace/studios/this_studio/PIPS/pips/dvae.py", line 518, in forward
    z = self.sample(log_alpha, tau) # [B, N, C]
  File "/teamspace/studios/this_studio/PIPS/pips/dvae.py", line 493, in sample
    tau = torch.as_tensor(temp, device=log_alpha.device, dtype=log_alpha.dtype)
```
- [x] DWKL scale look off. It has a scale >100x KL (The calculation was all messed up. Fixed it now)
- [x] Add ability to decide which training set to use
- [x] Move attributes like limit_training_samples and permute_train to experiment config
- [x] Compute the model code bits and compare with the dataset code bits
- [x] Make validation argmaxed (hard codes, no sampling or temperature)
- [x] Compute Codebook Usage as EMA
- [x] Add logits_tau monitoring and enable small initialisation.
- [x] Add position dependent head and codebook in GumbelCodebook (using position_dependent flag)
- [ ] Add logitlaplace loss ? (class LogitLaplace  in Karpathy's code)
- [ ] Change figure size for codebook usage and reconstructions
- [ ] Fix Monte Carlo KLD on compilation failure (unable to compile, seems some internal issue with torch)
- [ ] Do I need to jointly model discrete and continuous codes?

# Implement MNIST DVAE

## Training Tests
- [ ] Overfit on 32x1 batches (without betas and masking and no temperature and no hardness)
- [ ] Overfit on 32x10 batches (without betas and masking and no temperature and no hardness)
- [ ] Overfit on 100 batches (without betas and masking)
- [ ] Overfit on 100 batches (with masking and without betas)
- [ ] Overfit on 100 batches (with masking and betas)

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
- [x] Log serialised config to wandb
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


## Archived
- [ ] Dynamically adjust the beta? (It should be possible to do this by monitoring the rate of decay of the two loss components and adjusting the beta accordingly)
- [ ] Positional Embedding may be needed on top of the encoder output because all the latents come from the same codebook (Should not be needed as queries in pooler should double up as positional encodings)
- [ ] Codebook regularization (to bound it) (This can be done by adding weight decay to start with)
- [ ] Codebook Weight Initialisation
- [ ] Add evaluate CLI command


## Later
- [ ] Exponential Weighted Iterate Average for parameters (May be already in Pytorch Lightning) (Will use SWA instead)
- [ ] https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#stochastic-weight-averaging