import torch
from torchvision import transforms
from PIL import Image

# EXACT MODEL CLASS MATCH FROM TRAINING SCRIPT
class Proto1(torch.nn.Module):
    def __init__(self, img_size=256, num_classes=2):
        super().__init__()
        h = img_size // 8
        w = img_size // 8
        self.sequence_length = h * w

        # Stem
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.SiLU(),
        )

        self.stage2 = torch.nn.Sequential(
            MBConvBlock(64, 96, 4, stride=2),
            MBConvBlock(96, 96, 4, stride=1),
        )
        self.stage3 = torch.nn.Sequential(
            MBConvBlock(96, 192, 4, stride=2),
            MBConvBlock(192, 192, 4, stride=1),
        )
        self.to_sequence = torch.nn.Conv2d(192, 384, 1)
        self.pos_embedding = PositionalEmbedding(self.sequence_length, 384)
        self.transformer = torch.nn.Sequential(
            TransformerBlock(384),
            TransformerBlock(384),
        )
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.classifier = torch.nn.Linear(384, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.to_sequence(x)
        b, c, h, w = x.shape
        x = x.view(b, c, -1).transpose(1, 2)
        x = self.pos_embedding(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        return x

# --- Include the following building blocks, as defined in your model script ---

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, sequence_length, output_dim):
        super().__init__()
        self.pos_embedding = torch.nn.Parameter(torch.randn(1, sequence_length, output_dim))

    def forward(self, x):
        return x + self.pos_embedding

class SEBlock(torch.nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.squeeze = torch.nn.AdaptiveAvgPool2d(1)
        self.excitation = torch.nn.Sequential(
            torch.nn.Linear(channels, channels // reduction_ratio),
            torch.nn.SiLU(),
            torch.nn.Linear(channels // reduction_ratio, channels),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        s = self.squeeze(x).view(b, c)
        s = self.excitation(s).view(b, c, 1, 1)
        return x * s

class MBConvBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim, expand_ratio, stride=1):
        super().__init__()
        self.use_residual = stride == 1 and input_dim == output_dim
        hidden_dim = input_dim * expand_ratio
        self.expand = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim, hidden_dim, 1, bias=False),
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.SiLU(),
        )
        self.depthwise = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.SiLU(),
        )
        self.se = SEBlock(hidden_dim)
        self.project = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_dim, output_dim, 1, bias=False),
            torch.nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        residual = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.project(x)
        if self.use_residual:
            x = x + residual
        return x

class TransformerBlock(torch.nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attn = torch.nn.MultiheadAttention(dim, num_heads)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.GELU(),
            torch.nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x_t = x.transpose(0, 1)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        x = residual + attn_out.transpose(0, 1)
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        return x

# --- Preprocessing Function ---
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # Add normalization matching your training, if any.
        # transforms.Normalize(mean, std),
    ])
    return transform(image).unsqueeze(0)
