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

My plan is to implement a transformer based on dVAE, which uses a gumbel softmax based sampling to generate discrete latent states.