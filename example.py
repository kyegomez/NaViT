import torch
from navit.main import NaViT


n = NaViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    heads = 16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1,
    token_dropout_prob=0.1
)

images = [
    [torch.randn(3, 256, 256), torch.randn(3, 128, 128)],
    [torch.randn(3, 256, 256), torch.randn(3, 256, 128)],
    [torch.randn(3, 64, 256)]
]

preds = n(images)