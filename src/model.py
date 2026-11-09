import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SpectralChannelAttention(nn.Module):
 
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_mlp(F.adaptive_avg_pool2d(x, 1))
        max_out = self.shared_mlp(F.adaptive_max_pool2d(x, 1))
        scale = torch.sigmoid(avg_out + max_out)
        return x * scale


class RelationalCrossAttentionBlock(nn.Module):
 
    def __init__(self, dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        # feat_map: [B, C, H, W] -> tokens [B, H*W, C]
        b, c, h, w = feat_map.shape
        tokens = feat_map.flatten(2).transpose(1, 2)
        q_s = self.norm1(tokens)
        self_out, _ = self.self_attn(q_s, q_s, q_s)
        f_s = tokens + self_out

        # Global spectral context token for cross-attention
        context = tokens.mean(dim=1, keepdim=True).expand_as(tokens)
        q_c = self.norm2(f_s)
        cross_out, _ = self.cross_attn(q_c, context, context)
        f_c = f_s + cross_out

        combined = torch.cat([f_s.mean(dim=1), f_c.mean(dim=1)], dim=-1)
        return self.proj(combined)


class LaplacianHighFreqBranch(nn.Module):

    def __init__(self, out_dim: int = 128):
        super().__init__()
        # Fixed 3x3 Laplacian high-pass kernel
        lap_kernel = torch.tensor(
            [[[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]]
        ).unsqueeze(0)
        self.register_buffer("lap_weight", lap_kernel.repeat(3, 1, 1, 1))

        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        high_freq = F.conv2d(x, self.lap_weight, padding=1, groups=3)
        return self.conv_net(high_freq)


class CosineMarginClassifier(nn.Module):

    def __init__(self, in_features: int, num_classes: int, scale: float = 30.0, margin: float = 0.25):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        # Both embeddings and class centers are L2-normalized
        norm_emb = F.normalize(embeddings, p=2, dim=1)
        norm_weight = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(norm_emb, norm_weight).clamp(-1.0 + 1e-6, 1.0 - 1e-6)

        if labels is None or not self.training:
            return cosine * self.scale

        # Additive angular margin penalty during training
        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.margin)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        output = (one_hot * target_logits) + ((1.0 - one_hot) * cosine)
        return output * self.scale


class EnhancedSSAPeriocularNet(nn.Module):
    """
    - Head 1: 512-D L2-Normalized Metric Embedding + CosineMarginClassifier (Person ID)
    - Head 2: Domain-Robust Gender Classifier (Male=0, Female=1)
    - Head 3: 4-Attribute Occlusion & Worst-Case Classifier
              [Glasses, Hair_Occlusion, Eye_Closure, Extreme_Pose]
    """
    def __init__(self, num_identities: int = 344, backbone_name: str = "resnet34", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet34(weights=weights)
        else:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet18(weights=weights)

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4  # [B, 512, 7, 7]

        self.spectral_attn = SpectralChannelAttention(512, reduction=16)
        self.relational_encoder = RelationalCrossAttentionBlock(dim=512, num_heads=8)
        self.laplacian_branch = LaplacianHighFreqBranch(out_dim=128)

        # Fuse 512-D relational transformer features + 128-D Laplacian edge features -> 512-D embedding
        self.embedding_head = nn.Sequential(
            nn.Linear(512 + 128, 512, bias=False),
            nn.BatchNorm1d(512),
        )

        # Task 1: Person Identification (Closed-set logits + Open-set 512-D Cosine Embedding)
        self.id_classifier = CosineMarginClassifier(512, num_identities, scale=28.0, margin=0.20)

        # Task 2: Gender Classification Head
        self.gender_head = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

        # Task 3: Supervised Worst-Case / Occlusion Multi-Label Head
        # Predicts [Glasses, Hair_Occlusion, Eye_Closure, Pose_Deviation]
        self.occlusion_head = nn.Sequential(
            nn.Dropout(p=0.20),
            nn.Linear(512, 64),
            nn.GELU(),
            nn.Linear(64, 4),
        )

    def forward(self, x: torch.Tensor, id_labels: torch.Tensor = None):
        feat = self.stem(x)
        feat = self.layer1(feat)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        feat_map = self.layer4(feat)  # Target layer for Grad-CAM

        spec_map = self.spectral_attn(feat_map)
        rel_vec = self.relational_encoder(spec_map)
        lap_vec = self.laplacian_branch(x)

        raw_emb = self.embedding_head(torch.cat([rel_vec, lap_vec], dim=1))
        norm_emb = F.normalize(raw_emb, p=2, dim=1)

        id_logits = self.id_classifier(norm_emb, id_labels)
        gender_logits = self.gender_head(raw_emb)
        occlusion_logits = self.occlusion_head(raw_emb)

        return {
            "id_logits": id_logits,
            "gender_logits": gender_logits,
            "occlusion_logits": occlusion_logits,
            "embedding": norm_emb,
            "feat_map": spec_map,
        }