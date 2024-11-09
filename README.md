# Perception Informed Program Synthesis (PIPS)

In this research project, I want to focus on program synthesis given a predefined goal. (Setting of goals is not the focus of this project.)
The problem description is very much inspired by the ARC dataset set up where for each task, there is a few input/output pairs as demonstrations, which should 
be used to extract abstractions to synthesize a program that can generate the output for a new input.

One can indeed create a DSL and then bruteforce the search to find a program that can generate the output for a new input. However, the search space is often too large and the methods too slow due to combinatorial explosion. On the other hand, the deep learning models are highly data inefficient (here we only have 2-3 demonstrations for each task).

I independently arrived at the conclusion that we need a hybrid approach that can combine the benefits of deep learning and symbolic reasoning. Particularly, the neural networks can represent system 1 thinking (perception) and the symbolic reasoning can represent system 2 thinking (goal-based planning).

In this project, the goal is device a general framework that can combine the benefits of deep learning and symbolic reasoning.

To this end, I will use this readme file to record my thoughts and the progress of the project.

## 08 Nov 2024
Here's my key insight: We can use an autoencoder neural network to create compressed representations of both input and output grids. With these compressed representations, we can reframe program synthesis as finding a sequence of valid transformations that convert the input grid into the output grid.

Think of it like finding a path through a space of valid grid states. While we can't smoothly morph between grids (since they're discrete), we can find a sequence of valid intermediate grids that step from input to output. This effectively turns program synthesis into a pathfinding problem through the space of valid grid states.

As a first approach, we could imagine drawing a straight line from input to output in this representation space. We could then train a recurrent transformer model to find a sequence of valid grid states that follows this line as closely as possible, essentially finding the most direct valid path between input and output.

The key is finding good representations of the grids which captures independent attributes and making the search around a point more efficient. There is following research in this direction.

Here are some relevant autoencoding approaches that promote independent/disentangled representations:

1. β-VAE (Beta Variational Autoencoder)
   - Introduces a β parameter to balance reconstruction accuracy vs. disentanglement
   - Higher β values encourage more disentangled latent representations
   - Each dimension tends to capture independent factors of variation

2. Factor VAE
   - Adds a Total Correlation penalty term to encourage statistical independence between latent dimensions
   - Uses a discriminator to estimate the degree of factorization
   - Results in highly disentangled representations where each dimension corresponds to a meaningful factor

3. InfoGAN
   - Uses mutual information maximization between latent codes and generated outputs
   - Can learn interpretable representations without supervision
   - Different dimensions often capture distinct visual attributes like width, rotation, style etc.

4. DIP-VAE (Disentangled Inferred Prior VAE)
   - Matches the inferred latent distribution to a factorized prior
   - Promotes independence between different latent dimensions
   - Good at separating independent factors of variation

5. Spatial Broadcast Decoder
   - Specially designed decoder architecture that broadcasts latent vectors spatially
   - Makes it easier to learn position-invariant features
   - Can help separate spatial attributes (size, position) from other factors

The key advantages of these methods for our use case:
- They create structured latent spaces where different dimensions have clear semantic meaning
- Makes it easier to perform controlled edits by modifying specific dimensions
- Enables more efficient search and interpolation in the latent space
- Can help identify meaningful transformation operations between states

We can experiment with these different approaches to find which one creates the most useful representations for our grid-based program synthesis task. The ideal encoding would cleanly separate different visual attributes while maintaining enough structure to guide the search process.


Transformer-based models can also be effectively used for autoencoding:


1. Vision Transformer (ViT) Autoencoder
   - Splits image into patches and uses transformer encoder-decoder architecture
   - Self-attention helps capture global relationships between patches
   - Can learn powerful latent representations of visual data

2. VQGAN (Vector Quantized GAN with Transformers)
   - Combines transformer with vector quantization for discrete latent space
   - Particularly good at capturing high-level semantic features
   - Can generate high-quality reconstructions while maintaining semantic structure

3. MAE (Masked Autoencoder)
   - Randomly masks patches and tries to reconstruct them
   - Very effective self-supervised learning approach
   - Creates robust and meaningful latent representations

Key benefits of transformer-based autoencoders:
- Better at capturing long-range dependencies than CNNs
- Can handle variable-sized inputs naturally
- Strong inductive biases for modeling relationships between parts
- Scale well with more data and compute

These approaches could be particularly valuable for our grid-based task since they excel at modeling structural relationships and patterns across the entire input.
