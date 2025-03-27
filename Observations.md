# Observations

## Training

### Use pure logits for loss
- use_pure_logits_for_loss is not necessary for training accuracy

### Normalisation
- Using normalise_kq leads to less peaky distribution and renders diversity losses useless
- Normalise_kq + temp decay leads to high accuracy

### Disable Normalisation
- Delayed accuracy jump with temp decay
- Seems like it needs longer temperature decay to reach similar accuracy (no entropy loss needed)
- Interestingly the codebook collapses later, perhaps slower decay might fix that