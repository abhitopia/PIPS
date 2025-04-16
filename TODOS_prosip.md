- [x] Make activation function configurable
- [x] Introduce depth embedding for the interpreter
- [x] Figure out which BARC datasets to use
- [x] Create Small versions of the loaders
- [ ] Add option to limit loaded tasks in ArcTaskLoader


# Improvements
- [x] Each latent vector should get it's own sub-routine (use transformer to construct the sub-routine)
- [x] A state consists of a sequence of latent vectors. (It implicitly carries data as well as intermediate computations)
- [x] Subroutine Executor should take subroutine embeddings and previous state to generate next state
- [x] A next subroutine depends on the previous state and the iteration and the master program
- [x] The intermediate outputs are extracted from the state using output extractor
- [ ] Create MultiHeadAttention with RoPE
- [ ] A subroutine must come from a set of available sub-routines (use subroutine selector using embedding)
- [ ] Do proper initialization of the modules all throughout the code



interpreter(input, program, num_iterations) -> outputs
input2state(input, program, iteration) -> state
stateexecutor(state, subroutine_embedding) -> next_state
subroutine(program, iteration, state) -> next_subroutine