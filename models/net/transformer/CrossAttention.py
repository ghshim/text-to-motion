import torch
import torch.nn as nn

import torch
import torch.nn as nn
from timm.models.layers import DropPath


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv, attn_mask=None, padding_mask=None):
        B, N, C = x_kv.shape
        kv = (
            self.kv(x_kv)
            .reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = (
            kv[0],
            kv[1],
        )  # make torchscript happy (cannot use tensor as tuple)
        B, N_q, C = x_q.shape
        q = (
                self.q(x_q)
                .reshape(B, N_q, 1, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
        )[0]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask!=None:
            attn = attn+attn_mask

        if padding_mask!=None:
            attn = attn.masked_fill(padding_mask, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        attn_target,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_scale_type=None,  # from cait; possible values are None, "per_channel", "scalar"
        layer_scale_init_value=1e-4,  # from cait; float
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_q = norm_layer(dim)

        if isinstance(attn_target, nn.Module):
            self.attn = attn_target
        else:
            self.attn = attn_target(dim=dim)

        if drop_path > 0.0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.layer_scale_type = layer_scale_type

        # Layerscale
        if self.layer_scale_type is not None:
            assert self.layer_scale_type in [
                "per_channel",
                "scalar",
            ], f"Found Layer scale type {self.layer_scale_type}"
            if self.layer_scale_type == "per_channel":
                # one gamma value per channel
                gamma_shape = [1, 1, dim]
            elif self.layer_scale_type == "scalar":
                # single gamma value for all channels
                gamma_shape = [1, 1, 1]
            # two gammas: for each part of the fwd in the encoder
            self.layer_scale_gamma1 = nn.Parameter(
                torch.ones(size=gamma_shape) * layer_scale_init_value,
                requires_grad=True,
            )
            self.layer_scale_gamma2 = nn.Parameter(
                torch.ones(size=gamma_shape) * layer_scale_init_value,
                requires_grad=True,
            )

    def forward(self, x_q, x_kv, attn_mask=None, padding_mask=None):
        if self.layer_scale_type is None:
            x_q = x_q + self.drop_path(self.attn(self.norm1(x_q), self.norm1_q(x_kv), attn_mask=attn_mask, padding_mask=padding_mask))
            x_q = x_q + self.drop_path(self.mlp(self.norm2(x_q)))
        else:
            x_q = x_q + self.drop_path(self.attn(self.norm1(x_q, attn_mask=attn_mask, padding_mask=padding_mask)) * self.layer_scale_gamma1, self.norm1_q(x_kv)* self.layer_scale_gamma1)
            x_q = x_q + self.drop_path(self.mlp(self.norm2(x_q)) * self.layer_scale_gamma2)
        return x_q

    def extra_repr(self) -> str:
        named_modules = set()
        for p in self.named_modules():
            named_modules.update([p[0]])
        named_modules = list(named_modules)

        string_repr = ""
        for p in self.named_parameters():
            name = p[0].split(".")[0]
            if name not in named_modules:
                string_repr = (
                    string_repr
                    + "("
                    + name
                    + "): "
                    + "tensor("
                    + str(tuple(p[1].shape))
                    + ", requires_grad="
                    + str(p[1].requires_grad)
                    + ")\n"
                )

        return string_repr

class CrossTransformer(nn.Module):
    def __init__(self, attn_target,norm_layer, embed_dim, mlp_ratio=4,
                            layer_scale_type=None, layer_scale_init_value=1e-4,
                            depth=4,  drop_rate=0.1, drop_path_type = "progressive", drop_path_rate=0.1):
        super().__init__()


        self.crossattentionblocks = self.build_cross_transformer(attn_target=attn_target,norm_layer=norm_layer,
                                                                 embed_dim=embed_dim, mlp_ratio=mlp_ratio,
                                                                 layer_scale_type=layer_scale_type, layer_scale_init_value=layer_scale_init_value,
                                                                 depth=depth,  drop_rate=drop_rate, drop_path_type =drop_path_type, drop_path_rate=drop_path_rate)

    def forward(self, x_q, x_kv, attn_mask=None, padding_mask=None):
        for blk in self.crossattentionblocks:
            x = blk(x_q, x_kv, attn_mask=attn_mask, padding_mask=padding_mask)
        return x

    def build_cross_transformer(self, attn_target,norm_layer, embed_dim, mlp_ratio=4,
                                layer_scale_type=None, layer_scale_init_value=1e-4,
                                depth=4,  drop_rate=0.1, drop_path_type = "progressive", drop_path_rate=0.1):
        # Build Encoder Attention Blocks: stochastic depth decay rule
        if drop_path_type == "progressive":
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        elif drop_path_type == "uniform":
            dpr = [drop_path_rate for i in range(depth)]
        blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=embed_dim,
                    attn_target=attn_target,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    layer_scale_type=layer_scale_type,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for i in range(depth)
            ]
        )
        return blocks

if __name__ == '__main__':
    from functools import partial

    B,T,E,N = 256, 35, 128, 5
    patch_motion = torch.randn(B,N,E)
    patch_pose = torch.randn(B, 1, E)

    embed_dim, num_heads_encoder, nhid_encoder, drop_rate, depth_encoder, eps = E, 4, 64, 0.2, 4, 1.0e-8


    attn_target = partial(
        CrossAttention,
        attn_drop=0.1,
        num_heads=2,
        proj_drop=0.0,
        qk_scale=False,
        qkv_bias=True,
    )
    norm_layer = partial(nn.LayerNorm, eps=1.e-8)

    crosstransformer = CrossTransformer(attn_target, norm_layer, embed_dim, mlp_ratio=4,
                            layer_scale_type=None, layer_scale_init_value=1e-4,
                            depth=4,  drop_rate=0.1, drop_path_type = "progressive", drop_path_rate=0.1)

    print("patch_motion", patch_motion.shape)
    print("patch_pose", patch_pose.shape)
    x = crosstransformer(patch_pose, patch_motion)
    print("x", x.shape)
