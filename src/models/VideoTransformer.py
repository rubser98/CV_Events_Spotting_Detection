import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoTransformer(nn.Module):
    def __init__(self, clip_length, frame_height, frame_width, patch_size, num_classes, dim, depth, heads, mlp_dim):
        super(VideoTransformer, self).__init__()

        num_patches = (frame_height // patch_size) * (frame_width // patch_size)
        patch_dim = 3 * patch_size ** 2

        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, clip_length, num_patches + 1, dim))
        self.transformer_encoder = TransformerEncoder(dim, depth, heads, mlp_dim)
        self.classification_head = nn.Linear(dim * num_patches, num_classes)

    def forward(self, x):
        batch_size, clip_length, _, _, _ = x.size()
        x = x.reshape(batch_size * clip_length, 3, self.frame_height, self.frame_width)
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((self.positional_encoding[:, :clip_length].reshape(1, -1, self.dim), x), dim=1)
        x = self.transformer_encoder(x)
        x = x.flatten(2)
        x = self.classification_head(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.norm = nn.LayerNorm(dim)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.MultiheadAttention(dim, heads),
                nn.Linear(dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, dim),
            ]))

    def forward(self, x):
        for attention, linear_1, activation, linear_2 in self.layers:
            x = x + attention(self.norm(x))[0]
            x = x + linear_2(activation(linear_1(self.norm(x))))
        return x
