- [x] Make activation function configurable
- [x] Introduce depth embedding for the interpreter
- [x] Figure out which BARC datasets to use
- [x] Create Small versions of the loaders
- [ ] Add option to limit loaded tasks in ArcTaskLoader


# Improvements
- [ ] A state consists of a sequence of latent vectors. (It implicitly carries data as well as intermediate computations)
- [ ] Subroutine Executor should take subroutine embedding and previous state to generate next state
- [ ] Each latent vector should get it's own sub-routine (use transformer to construct the sub-routine)
- [ ] A next subroutine depends on the previous state and the iteration and the master program
- [ ] The intermediate outputs are extracted from the state using output extractor
- [ ] A subroutine must come from a set of available sub-routines (use subroutine selector using embedding)