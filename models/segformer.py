# basic imports
import numpy as np

# DL library imports
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as pth_transforms

from einops import rearrange
from timm.models.layers import drop_path, trunc_normal_

class overlap_patch_embed(nn.Module):
    def __init__(self, patch_size, stride, in_chans, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size // 2, patch_size // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return x, h, w

class efficient_self_attention(nn.Module):

    def __init__(self, attn_dim, num_heads, dropout_p, sr_ratio, return_attention=False):
        super().__init__()
        assert attn_dim % num_heads == 0, f'expected attn_dim {attn_dim} to be a multiple of num_heads {num_heads}'
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(attn_dim, attn_dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(attn_dim)

        # Multi-head Self-Attention using dot product
        # Query - Key Dot product is scaled by root of head_dim
        self.q = nn.Linear(attn_dim, attn_dim, bias=True)
        self.kv = nn.Linear(attn_dim, attn_dim * 2, bias=True)
        self.scale = (attn_dim // num_heads) ** -0.5

        # Projecting concatenated outputs from
        # multiple heads to single `attn_dim` size
        self.proj = nn.Linear(attn_dim, attn_dim)

        self._return_attention = return_attention

    @property
    def return_attention(self):
        # Read-only access to the return_attention flag
        return self._return_attention

    def forward(self, x, h, w):
        q = self.q(x)
        q = rearrange(q, ('b hw (m c) -> b m hw c'), m=self.num_heads)

        if self.sr_ratio > 1:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            x = self.sr(x)
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.norm(x)

        x = self.kv(x)
        x = rearrange(x, 'b d (a m c) -> a b m d c', a=2, m=self.num_heads)
        k, v = x[0], x[1] # x.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = attn @ v
        x = rearrange(x, 'b m hw c -> b hw (m c)')
        x = self.proj(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        attn_output = {'key' : k, 'query' : q, 'value' : v, 'attn' : attn}
        if self.return_attention:
            return x, attn_output
        else:
            return x


class mix_feedforward(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, dropout_p = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

        # Depth-wise separable convolution
        self.conv = nn.Conv2d(hidden_features, hidden_features, (3, 3), padding=(1, 1),
                              bias=True, groups=hidden_features)
        self.dropout_p = dropout_p


    def forward(self, x, h, w):
        x = self.fc1(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.conv(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x


class transformer_block(nn.Module):
    def __init__(self, dim, num_heads, dropout_p, drop_path_p, sr_ratio,return_attention=False):
        super().__init__()
        # One transformer block is defined as :
        # Norm -> self-attention -> Norm -> FeedForward
        # skip-connections are added after attention and FF layers
        self.attn = efficient_self_attention(attn_dim=dim, num_heads=num_heads,
                    dropout_p=dropout_p, sr_ratio=sr_ratio,return_attention=return_attention)
        self.ffn = mix_feedforward( dim, dim, hidden_features=dim * 4, dropout_p=dropout_p)

        self.drop_path_p = drop_path_p
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self._return_attention = return_attention

    @property
    def return_attention(self):
        # Read-only access to the return_attention flag
        return self._return_attention


    def forward(self, x, h, w):
        # Norm -> self-attention
        skip = x
        x = self.norm1(x)
        if(self.return_attention):
            x, attn_output = self.attn(x, h, w)
        else:
            x = self.attn(x, h, w)
        x = drop_path(x, drop_prob=self.drop_path_p, training=self.training)
        x = x + skip

        # Norm -> FeedForward
        skip = x
        x = self.norm2(x)
        x = self.ffn(x, h, w)
        x = drop_path(x, drop_prob=self.drop_path_p, training=self.training)
        x = x + skip
        return x, attn_output

class mix_transformer_stage(nn.Module):
    def __init__(self, patch_embed, blocks, norm):
        super().__init__()
        self.patch_embed = patch_embed
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm

    def forward(self, x):
        x, h, w = self.patch_embed(x)
        for block in self.blocks:
            x = block(x, h, w)
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x

class mix_transformer_stage(nn.Module):
    def __init__(self, patch_embed, blocks, norm, return_attention=False):
        super().__init__()
        self.patch_embed = patch_embed
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm

        self._return_attention = return_attention

    @property
    def return_attention(self):
        # Read-only access to the return_attention flag
        return self._return_attention

    def forward(self, x):
        # patch embedding and store required data
        stage_output  = {}
        if(self.return_attention):
            stage_output['patch_embed_input'] = x
            x, h, w = self.patch_embed(x)
            stage_output['patch_embed_h'] = h
            stage_output['patch_embed_w'] = w
            stage_output['patch_embed_output'] = x

            for block in self.blocks:
                x, attn_output = block(x, h, w)
        else:
             x, h, w = self.patch_embed(x)
             for block in self.blocks:
                 x = block(x, h, w)

        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        # store last attention block data
        # in stages' output data
        if(self.return_attention):
            for k,v in attn_output.items():
                stage_output[k] = v
            del attn_output
            return x, stage_output
        else:
            return x


class mix_transformer(nn.Module):
    def __init__(self, in_chans, embed_dims, num_heads, depths,
                sr_ratios, dropout_p, drop_path_p,return_attention=False):
        super().__init__()
        self.stages = nn.ModuleList()
        for stage_i in range(len(depths)):
            # Each Stage consists of following blocks :
            # Overlap patch embedding -> mix_transformer_block -> norm
            blocks = []
            for i in range(depths[stage_i]):
                blocks.append(transformer_block(dim = embed_dims[stage_i],
                        num_heads= num_heads[stage_i], dropout_p=dropout_p,
                        drop_path_p = drop_path_p * (sum(depths[:stage_i])+i) / (sum(depths)-1),
                        sr_ratio = sr_ratios[stage_i],return_attention=return_attention))

            if(stage_i == 0):
                patch_size = 7
                stride = 4
                in_chans = in_chans
            else:
                patch_size = 3
                stride = 2
                in_chans = embed_dims[stage_i -1]

            patch_embed = overlap_patch_embed(patch_size, stride=stride, in_chans=in_chans,
                            embed_dim= embed_dims[stage_i])
            norm = nn.LayerNorm(embed_dims[stage_i], eps=1e-6)
            self.stages.append(mix_transformer_stage(patch_embed, blocks, norm, return_attention=return_attention))

            self._return_attention = return_attention

    @property
    def return_attention(self):
        # Read-only access to the return_attention flag
        return self._return_attention


    def forward(self, x):
        outputs = []
        for i,stage in enumerate(self.stages):
            x, _ = stage(x)
            outputs.append(x)
        return outputs


    def get_attn_outputs(self, x):
        stage_outputs = []
        for i,stage in enumerate(self.stages):
            x, stage_data = stage(x)
            stage_outputs.append(stage_data)
        return stage_outputs


class segformer_head(nn.Module):
    def __init__(self, in_channels, num_classes, embed_dim, dropout_p=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.dropout_p = dropout_p

        # 1x1 conv to fuse multi-scale output from encoder
        self.layers = nn.ModuleList([nn.Conv2d(chans, embed_dim, (1, 1))
                                     for chans in reversed(in_channels)])
        self.linear_fuse = nn.Conv2d(embed_dim * len(self.layers), embed_dim, (1, 1), bias=False)
        self.bn = nn.BatchNorm2d(embed_dim, eps=1e-5)

        # 1x1 conv to get num_class channel predictions
        self.linear_pred = nn.Conv2d(self.embed_dim, num_classes, kernel_size=(1, 1))
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.linear_fuse.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        feature_size = x[0].shape[2:]

        # project each encoder stage output to H/4, W/4
        x = [layer(xi) for layer, xi in zip(self.layers, reversed(x))]
        x = [F.interpolate(xi, size=feature_size, mode='bilinear', align_corners=False)
             for xi in x[:-1]] + [x[-1]]

        # concatenate project output and use 1x1
        # convs to get num_class channel output
        x = self.linear_fuse(torch.cat(x, dim=1))
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.linear_pred(x)
        return x



class segformer_mit_b3(nn.Module):
    def __init__(self, in_channels, num_classes, return_attention=False):
        super().__init__()
        # Encoder block
        self.backbone = mix_transformer(in_chans=in_channels, embed_dims=(64, 128, 320, 512),
                                    num_heads=(1, 2, 5, 8), depths=(3, 4, 18, 3),
                                    sr_ratios=(8, 4, 2, 1), dropout_p=0.0, drop_path_p=0.1,return_attention=return_attention)
        # decoder block
        self.decoder_head = segformer_head(in_channels=(64, 128, 320, 512),
                                    num_classes=num_classes, embed_dim=256)

        # init weights
        self.apply(self._init_weights)
        self._return_attention = return_attention

    @property
    def return_attention(self):
        # Read-only access to the return_attention flag
        return self._return_attention

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")


    def forward(self, x):
        image_hw = x.shape[2:]
        x = self.backbone(x)
        x = self.decoder_head(x)
        x = F.interpolate(x, size=image_hw, mode='bilinear', align_corners=False)
        return x


    def get_attention_outputs(self, x):
        if(self.return_attention):
            return self.backbone.get_attn_outputs(x)
        else:
            raise ValueError("Retrieving Attention map not enabled")


    def get_last_selfattention(self, x):
        if(self.return_attention):
            outputs = self.get_attention_outputs(x)
            return outputs[-1].get('attn', None)
        else:
            raise ValueError("Retrieving Attention map not enabled")
