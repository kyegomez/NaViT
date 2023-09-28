from functools import partial
from typing import List, Union

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence as orig_pad_sequence
from zeta.nn import FeedForward, FlashAttention, LayerNorm, RMSNorm
from zeta.utils import always, default, divisible_by, exists, pair
from mgqa.attention import MGQA as Attention
from einops import rearrange, reduce, repeat
import torch.nn.functional as F

#utils
def group_images_by_max_seq_len(
    images: List[Tensor],
    patch_size: int,
    calc_token_dropout = None,
    max_seq_len = None
) -> List[List[Tensor]]:
    
    calc_token_dropout = default(
        calc_token_dropout, always(0.)
    )

    groups = []
    group = []
    seq_len = 0

    if isinstance(
        calc_token_dropout,
        (float, int)
    ):
        calc_token_dropout = always(calc_token_dropout)
        
    for image in images:
        assert isinstance(image, Tensor)

        image_dims = image.shape[-2:]
        ph, pw = map(
            lambda t: t // patch_size, image_dims
        )

        image_seq_len = (ph * pw) + 1
        image_seq_len = int(
            image_seq_len * (1 - calc_token_dropout(*image_dims))
        )

        assert image_seq_len <= max_seq_len, f"Image with dimensions {image_dims} exceeds maximum sequence length"

        if (seq_len + image_seq_len) > max_seq_len:
            groups.append(group)
            group = []
            seq_len = 0
        
        group.append(image)
        seq_len += image_seq_len

    if len(group) > 0:
        groups.append(group)
    
    return groups


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(
                    dim,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout
                ),
                FeedForward(
                    dim,
                    mlp_dim,
                    dropout=dropout
                )
            ]))

        self.norm = LayerNorm(dim)
    
    def forward(
        self,
        x,
        mask=None,
        attn_mask=None
    ):
        for attn, ff in self.layers:
            x = attn(
                x,
                mask=mask,
                attn_mask=attn_mask
            ) + x
            x = ff(x) + x
        
        return self.norm(x)
    

class NaViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64,
        dropout=0.,
        emb_dropout=0.,
        token_dropout_prob=None
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        
        self.calc_token_dropout = None

        if callable(token_dropout_prob):
            self.calc_token_dropout = token_dropout_prob
        
        elif isinstance(
            token_dropout_prob,
            (float, int)
        ):
            assert 0. < token_dropout_prob < 1.
            token_dropout_prob = always(token_dropout_prob)
            self.calc_token_dropout = token_dropout_prob

        #calculate patching related stuff
        assert divisible_by(
            image_height,
            patch_size
        ) and divisible_by(
            image_width, patch_size
        ), 'Image dimensions must be divisible by patch size'

        patch_height_dim, patch_width_dim = (image_height // patch_size), (image_width // patch_size)
        patch_dim = channels * (patch_size ** 2)

        self.channels = channels
        self.patch_size = patch_size

        self.to_patch_embedding = nn.Sequential(
            LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            LayerNorm(dim),
        )

        self.pos_embed_height = nn.Parameter(
            torch.randn(patch_height_dim, dim)
        )
        self.pos_embed_width = nn.Parameter(
            torch.randn(patch_width_dim, dim)
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout
        )

        #final attention pooling queries
        self.attn_pool_queries = nn.Parameter(torch.randn(dim))
        self.attn_pool = Attention(
            dim=dim,
            dim_head=dim_head,
            heads=heads
        )

        #output to logits
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(
                dim,
                num_classes,
                bias=False
            )
        )

    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(
        self,
        batched_images: Union[List[Tensor], List[List[Tensor]]],
        group_images = False,
        group_max_seq_len = 2048
    ):
        p, c, device, has_token_dropout = self.patch_size, self.channels, self.device, exists(self.calc_token_dropout)

        arange = partial(
            torch.arange, 
            device=device
        )

        pad_sequence = partial(
            orig_pad_sequence,
            batch_first=True
        )

        #auto pack if specified
        if group_images:
            batch_images = group_images_by_max_seq_len(
                batched_images,
                patch_size=self.patch_size,
                calc_token_dropout=self.calc_token_dropout,
                max_seq_len=group_max_seq_len
            )
        
        #varibale lengthed seqs with attention masks
        num_images = []
        batched_sequences = []
        batched_positions = []
        batched_image_ids = []

        for images in batched_images:
            num_images.append(len(images))

            sequences = []
            positions = []
            image_ids = torch.empty(
                (0,),
                device=device,
                dtype=torch.long
            )

            for image_id, image in enumerate(images):
                assert image.ndim == 3 and image.shape[0] == c
                image_dims = image.shape[-2:]
                assert all(
                    [divisible_by(dim, p) for dim in image_dims]
                ), f"Height and width {image_dims} of images must be divisible by patch_size {p}"

                ph, pw = map(lambda dim: dim // p, image_dims)

                pos = torch.stack(
                    torch.meshgrid((
                        arange(ph),
                        arange(pw)
                    ), indexing='ij'), dim=-1
                )

                pos = rearrange(
                    pos,
                    'h w c -> (h w) c'
                )
                seq = rearrange(
                    image,
                    'c (h p1) (w p2) -> (h w) (c p1 p2)',
                    p1=p,
                    p2=p
                )

                seq_len = seq.shape[-2]

                if has_token_dropout:
                    token_dropout = self.calc_token_dropout(*image_dims)
                    num_keep = max(
                        1,
                        int(seq_len * (1 - token_dropout))
                    )
                    keep_indices = torch.randn(
                        (seq_len,),
                        device = device
                    ).topk(num_keep, dim=-1).indices

                    seq = seq[keep_indices]
                    pos = pos[keep_indices]

                image_ids = F.pad(
                    image_ids,
                    (
                        0, 
                        seq.shape[-2]
                    ), value=image_id
                )

                sequences.append(seq)
                positions.append(pos)
            
            batched_image_ids.append(image_ids)
            batched_sequences.append(torch.cat(sequences, dim=0))
            batched_positions.append(torch.cat(positions, dim=0))
        
        lengths = torch.tensor(
            [seq.shape[-2] for seq in batched_sequences],
            device=device,
            dtype=torch.long
        )
        max_lengths = arange(lengths.amax().item()), 
        key_pad_mask = rearrange(
            lengths,
            'b -> b 1'
        ) <= rearrange(max_lengths, 'n -> 1 n')

        #derive attention mask, and combine key padding mask from above
        batched_image_ids = pad_sequence(batched_image_ids)
        attn_mask = rearrange(
            batched_image_ids,
            'b i -> b 1 i 1'
        ) == rearrange(
            batched_image_ids,
            'b j -> b 1 1 j'
        )
        attn_mask = attn_mask & rearrange(
            key_pad_mask,
            'b j -> b 1 1 j'
        )

        #combine patched as well as the patched width / height positions for 2d positional embedding
        patches = pad_sequence(batched_sequences)
        patch_positions = pad_sequence(batched_positions)

        #need to how many images for final attention pooling
        num_images = torch.tensor(
            num_images,
            device=device,
            dtype=torch.long
        )

        #patchify
        x = self.to_patch_embedding(patches)

        #factorized 2d absolute positional embedding
        h_indices, w_indices = patch_positions.unbind(dim=-1)

        h_pos = self.pos_embed_height[h_indices]
        w_pos = self.pos_embed_width[w_indices]

        x = x + h_pos + w_pos

        x = self.dropout(x)

        #attention
        x = self.transformer(x, attn_mask=attn_mask)

        #do attention pooling at the end
        max_queries = num_images.amax().item()

        queries = repeat(
            self.attn_pool_queries,
            'd -> b n d',
            n=max_queries,
            b=x.shape[0]
        )

        #pool mask
        image_id_arange = arange(max_queries)
        attn_pool_mask = rearrange(
            image_id_arange,
            'i -> i 1',
        ) == rearrange(
            batched_image_ids,
            'b j -> b 1 j'
        )
        attn_pool_mask = attn_pool_mask & rearrange(
            key_pad_mask,
            'b j -> b 1 j'
        )
        attn_pool_mask = rearrange(
            attn_pool_mask,
            'b i j - b 1 i j'
        )

        #attention pool
        x = self.attn_pool(
            queries,
            context=x,
            attn_mask=attn_pool_mask,
        ) + queries
        x = rearrange(
            x,
            'b n d -> (b n) d'
        )

        #each batch element may not have same images
        is_images = image_id_arange < rearrange(
            num_images,
            "b -> b 1"
        )
        is_images = rearrange(
            is_images,
            'b n -> (b  n)'
        )

        x = x[is_images]

        #project to logits
        x = self.to_latent(x)
        
        return self.mlp_head(x)

