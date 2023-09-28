[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# NaViT
My implementation of "Patch n’ Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution"

[Paper Link](https://arxiv.org/pdf/2307.06304.pdf)

# Appreciation
* Lucidrains
* Agorians

# Install
`pip install navit-torch`

# Usage
```python
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
```

# Dataset Strategy
Here is a table of the key datasets and their metadata used for pretraining and evaluating NaViT:

| Dataset | Type | Size | Details | Source |  
|-|-|-|-|-|
| JFT-4B | Image classification | 4 billion images | Private dataset from Google | [1] |
| WebLI | Image-text | 73M image-text pairs | Web-crawled dataset | [2] |
| ImageNet | Image classification | 1.3M images, 1000 classes | Standard benchmark | [3] |
| ImageNet-A | Image classification | 7,500 images | Out-of-distribution variant | [4] |  
| ObjectNet | Image classification | 50K images, 313 classes | Out-of-distribution variant | [5] |
| LVIS | Object detection | 120K images, 1000 classes | Large vocabulary instance segmentation | [6] |
| ADE20K | Semantic segmentation | 20K images, 150 classes | Scene parsing dataset | [7] |
| Kinetics-400 | Video classification | 300K videos, 400 classes | Action recognition dataset | [8] |
| FairFace | Face attribute classification | 108K images, 9 attributes | Balanced dataset for facial analysis | [9] |
| CelebA | Face attribute classification | 200K images, 40 attributes | Face attributes dataset | [10] |

[1] Zhai et al. "Scaling Vision Transformers". 2022. https://arxiv.org/abs/2106.04560  
[2] Chen et al. "PaLI". 2022. https://arxiv.org/abs/2209.06794
[3] Deng et al. "ImageNet". 2009. http://www.image-net.org/
[4] Hendrycks et al. "Natural Adversarial Examples". 2021. https://arxiv.org/abs/1907.07174
[5] Barbu et al. "ObjectNet". 2019. https://arxiv.org/abs/1612.03916
[6] Gupta et al. "LVIS". 2019. https://arxiv.org/abs/1908.03195 
[7] Zhou et al. "ADE20K". 2017. https://arxiv.org/abs/1608.05442
[8] Kay et al. "Kinetics". 2017. https://arxiv.org/abs/1705.06950
[9] Kärkkäinen and Joo. "FairFace". 2019. https://arxiv.org/abs/1908.04913
[10] Liu et al. "CelebA". 2015. https://arxiv.org/abs/1410.5408

# Todo
- create example trainining script

# License
MIT

# Citations
```
@misc{2307.06304,
Author = {Mostafa Dehghani and Basil Mustafa and Josip Djolonga and Jonathan Heek and Matthias Minderer and Mathilde Caron and Andreas Steiner and Joan Puigcerver and Robert Geirhos and Ibrahim Alabdulmohsin and Avital Oliver and Piotr Padlewski and Alexey Gritsenko and Mario Lučić and Neil Houlsby},
Title = {Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution},
Year = {2023},
Eprint = {arXiv:2307.06304},
}
```
