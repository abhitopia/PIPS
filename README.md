# Perception Informed Program Synthesis (PIPS)

## DiscreteVAE (DVAE)

### Modes of Operation
    
1. **Sampling, Approximate KLD Loss**
    - Uses Gumbel Noise and relaxed (temperature controlled) softmax.
    - The temperature is annealed from a high value to a low value.
    - The approximate KLD loss is **independent** of the temperature.

2. **Sampling, Monte Carlo KLD Loss**
    - Uses Gumbel Noise and relaxed (temperature controlled) softmax.
    - The temperature is annealed from a high value to a low value.
    - The Monte Carlo KLD loss is **dependent** on the temperature.

3. **No Sampling, Approximate KLD Loss**
    - Uses regular softmax
    - The peakiness of the softmax is **dependent** on the temperature.
    - The approximate KLD loss is **independent** of the temperature.
    
    **Note**: If temperature is kept constant at 1, then regular softmax is used. KLD loss may or may not be used.

*Note: The Monte Carlo KLD loss is not available in this mode.*
