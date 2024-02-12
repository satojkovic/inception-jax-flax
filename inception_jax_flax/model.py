from flax import linen as nn


class InceptionBlock(nn.Module):
    @nn.compact
    def __call__(self, x, train=True):
        pass
