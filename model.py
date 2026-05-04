import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, window_size):
    B, D, H, W, C = x.shape

    x = x.view(
        B,
        D // window_size,
        window_size,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C
    )

    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, window_size ** 3, C)

    return windows


def window_reverse(windows, window_size, B, D, H, W, C):
    x = windows.view(
        B,
        D // window_size,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        window_size,
        C
    )

    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, D, H, W, C)

    return x


class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels=1, embed_dim=32, patch_size=4):
        super().__init__()

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm(x)
        return x


class WindowAttention3D(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.last_attention = None

    def forward(self, x):
        B_, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        self.last_attention = attn.detach()

        out = attn @ v
        out = out.transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)

        return out


class SwinBlock3D(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=4, shift_size=0, mlp_ratio=2.0):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, num_heads)

        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        B, D, H, W, C = x.shape
        shortcut = x

        x = self.norm1(x)

        if self.shift_size > 0:
            x = torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size, -self.shift_size),
                dims=(1, 2, 3)
            )

        windows = window_partition(x, self.window_size)
        attn_windows = self.attn(windows)

        x = window_reverse(
            attn_windows,
            self.window_size,
            B, D, H, W, C
        )

        if self.shift_size > 0:
            x = torch.roll(
                x,
                shifts=(self.shift_size, self.shift_size, self.shift_size),
                dims=(1, 2, 3)
            )

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        attention_info = {
            "attention": self.attn.last_attention,
            "grid_size": (D, H, W),
            "window_size": self.window_size,
            "shift_size": self.shift_size
        }

        return x, attention_info


class PatchMerging3D(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.reduction = nn.Linear(dim * 8, dim * 2)
        self.norm = nn.LayerNorm(dim * 8)

    def forward(self, x):
        B, D, H, W, C = x.shape

        if D % 2 != 0 or H % 2 != 0 or W % 2 != 0:
            x = x[:, :D - D % 2, :H - H % 2, :W - W % 2, :]
            B, D, H, W, C = x.shape

        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class LightweightSwin3D(nn.Module):
    def __init__(
        self,
        in_channels=1,
        embed_dim=32,
        num_heads=(2, 4, 8),
        window_size=4
    ):
        super().__init__()

        self.patch_embed = PatchEmbed3D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=4
        )

        self.stage1 = nn.ModuleList([
            SwinBlock3D(embed_dim, num_heads[0], window_size, shift_size=0),
            SwinBlock3D(embed_dim, num_heads[0], window_size, shift_size=2)
        ])

        self.merge1 = PatchMerging3D(embed_dim)

        self.stage2 = nn.ModuleList([
            SwinBlock3D(embed_dim * 2, num_heads[1], window_size, shift_size=0),
            SwinBlock3D(embed_dim * 2, num_heads[1], window_size, shift_size=2)
        ])

        self.merge2 = PatchMerging3D(embed_dim * 2)

        self.stage3 = nn.ModuleList([
            SwinBlock3D(embed_dim * 4, num_heads[2], window_size, shift_size=0)
        ])

        self.norm = nn.LayerNorm(embed_dim * 4)
        self.head = nn.Linear(embed_dim * 4, 1)

    def forward(self, x, return_attention=False):
        attention_maps = []

        x = self.patch_embed(x)

        for block in self.stage1:
            x, attn = block(x)
            attention_maps.append(attn)

        x = self.merge1(x)

        for block in self.stage2:
            x, attn = block(x)
            attention_maps.append(attn)

        x = self.merge2(x)

        for block in self.stage3:
            x, attn = block(x)
            attention_maps.append(attn)

        x = self.norm(x)
        x = x.mean(dim=(1, 2, 3))

        logits = self.head(x).squeeze(1)

        if return_attention:
            return logits, attention_maps

        return logits