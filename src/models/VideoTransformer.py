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
