import torch
import torch.nn as nn
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def get_fourier_embeds_from_boundingbox(embed_dim, box):
    """
    Args:
        embed_dim: int
        box: a 3-D tensor [B x N x 4] representing the bounding boxes for GLIGEN pipeline
    Returns:
        [B x N x embed_dim] tensor of positional embeddings
    """

    batch_size, num_boxes = box.shape[:2]

    emb = 100 ** (torch.arange(embed_dim) / embed_dim)
    emb = emb[None, None, None].to(device=box.device, dtype=box.dtype)
    emb = emb * box.unsqueeze(-1)

    emb = torch.stack((emb.sin(), emb.cos()), dim=-1)
    emb = emb.permute(0, 1, 3, 4, 2).reshape(batch_size, num_boxes, embed_dim * 2 * 4)

    return emb

class PixArtAlphaTextProjection(nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        elif act_fn == "silu_fp32":
            self.act_1 = FP32SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class ObjectLayoutEncoder(nn.Module):
    def __init__(self, positive_len, out_dim, fourier_freqs=8 ,max_boxes_token_length=30):
        super().__init__()
        self.positive_len = positive_len
        self.out_dim = out_dim

        self.fourier_embedder_dim = fourier_freqs
        self.position_dim = fourier_freqs * 2 * 4  # 2: sin/cos, 4: xyxy #64

        if isinstance(out_dim, tuple):
            out_dim = out_dim[0]

        self.null_positive_feature = torch.nn.Parameter(torch.zeros([max_boxes_token_length, self.positive_len]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))
        
    
        self.linears = PixArtAlphaTextProjection(in_features=self.positive_len + self.position_dim,hidden_size=out_dim//2,out_features=out_dim, act_fn="silu")

    def forward(
            self,
            boxes,  # [B,10,4]
            masks,  # [B,10]
            positive_embeddings,  # [B,10,30,3072]
        ):
        
        B, N, S, C = positive_embeddings.shape  # B: batch_size, N: 10, S: 30, C: 3072
        
        positive_embeddings = positive_embeddings.reshape(B*N, S, C)  # [B*10,30,3072]
        masks = masks.reshape(B*N, 1, 1)  # [B*10,1,1]
        
        # Process positional encoding
        xyxy_embedding = get_fourier_embeds_from_boundingbox(self.fourier_embedder_dim, boxes)  # [B,10,64]
        xyxy_embedding = xyxy_embedding.reshape(B*N, -1)  # [B*10,64]
        xyxy_null = self.null_position_feature.view(1, -1)  # [1,64]
        
        # Expand positional encoding to match sequence dimension
        xyxy_embedding = xyxy_embedding.unsqueeze(1).expand(-1, S, -1)  # [B*10,30,64]
        xyxy_null = xyxy_null.unsqueeze(0).expand(B*N, S, -1)  # [B*10,30,64]
        
        # Apply mask
        xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null  # [B*10,30,64]
        
        # Process feature encoding
        positive_null = self.null_positive_feature.view(1, S, -1).expand(B*N, -1, -1)  # [B*10,30,3072]
        positive_embeddings = positive_embeddings * masks + (1 - masks) * positive_null  # [B*10,30,3072]
        
        # Concatenate positional encoding and feature encoding
        combined = torch.cat([positive_embeddings, xyxy_embedding], dim=-1)  # [B*10,30,3072+64]
        
        # Process each box's features independently
        objs = self.linears(combined)  # [B*10,30,3072]
        
        # Restore original shape
        objs = objs.reshape(B, N, S, -1)  # [B,10,30,3072]
        
        return objs
    
class ObjectLayoutEncoder_noFourier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linears = PixArtAlphaTextProjection(in_features=self.in_dim,hidden_size=out_dim//2,out_features=out_dim, act_fn="silu")

    def forward(
            self,
            positive_embeddings,  # [B,10,30,3072]
        ):
        
        B, N, S, C = positive_embeddings.shape  # B: batch_size, N: 10, S: 30, C: 3072
        positive_embeddings = positive_embeddings.reshape(B*N, S, C)  # [B*10,30,3072]
        
        # Process each box's features independently
        objs = self.linears(positive_embeddings)  # [B*10,30,3072]
        
        # Restore original shape
        objs = objs.reshape(B, N, S, -1)  # [B,10,30,3072]
        
        return objs