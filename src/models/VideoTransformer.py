import torch
import torch.nn as nn

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, num_heads, num_layers):
        super(ViT, self).__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size[1] // patch_size[1]) * (image_size[2] // patch_size[2])
        self.patch_dim = image_size[0] * patch_size[1] * patch_size[2]
        
        self.patch_embedding = nn.Conv2d(image_size[0], dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, num_heads),
            num_layers
        )
        self.classifier = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        patches = self.patch_embedding(x)
        batch_size = patches.shape[0]
        patches = patches.flatten(2).permute(0, 2, 1)
        
        patches = torch.cat([torch.zeros(batch_size, 1, self.patch_dim).to(x.device), patches], dim=1)
        patches += self.positional_embedding
        
        encoded_patches = self.transformer_encoder(patches)
        representation = encoded_patches[:, 0]
        
        logits = self.classifier(representation)
        return logits.view(batch_size, -1)




import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionTransformer(nn.Module):
    def __init__(self, input_shape, output_shape, patch_size, num_classes, hidden_dim, num_heads, num_layers):
        super(VisionTransformer, self).__init__()

        num_patches = (input_shape[-2] // patch_size) * (input_shape[-1] // patch_size)
        patch_dim = input_shape[1] * patch_size * patch_size

        self.patch_embedding = nn.Conv2d(input_shape[1], hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim),
            num_layers
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2).permute(0, 2, 1)
        batch_size = x.size(0)
        x = torch.cat([x, self.positional_encoding.repeat(batch_size, 1, 1)], dim=1)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)

        return x.view(self.output_shape)


# Define the model with desired parameters
input_shape = (10, 27, 128, 320)
output_shape = (10, 448)
patch_size = 16
num_classes = 448
hidden_dim = 256
num_heads = 8
num_layers = 6

model = VisionTransformer(input_shape, output_shape, patch_size, num_classes, hidden_dim, num_heads, num_layers)
