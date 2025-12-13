# metai/model/mamba/module.py

import torch
import torch.nn as nn
from timm.layers.drop import DropPath
from timm.layers.weight_init import trunc_normal_

from mamba_ssm import Mamba

class TokenSpaceMLP(nn.Module):
    """
    Token Space MLP: 
    专为 [B, L, C] 格式设计的 MLP，替代了 SimVP 中基于 Conv2d 的 MixMlp。
    这使得 Mamba 模块不再依赖 SimVP 的基础层定义。
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # Input shape: [B, L, C]
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AdaptiveFusionGate(nn.Module):
    """
    Adaptive Fusion Gate:
    用于融合水平和垂直扫描特征的门控机制。
    在此独立实现，避免引用外部代码。
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        gate_dim = max(dim // reduction, 8)
        
        self.gate_net = nn.Sequential(
            nn.Linear(dim, gate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gate_dim, dim),
            nn.Sigmoid()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def forward(self, h_feat, v_feat):
        """
        Args:
            h_feat: [B, H, W, C] 水平扫描特征
            v_feat: [B, H, W, C] 垂直扫描特征
        """
        # (h + v) / 2 -> [B, H, W, C] -> Global Average Pooling -> [B, C]
        combined_context = (h_feat + v_feat) / 2.0
        context = combined_context.mean(dim=(1, 2)) 
        
        # 生成门控权重 [B, 1, 1, C] 以便广播
        gate = self.gate_net(context).unsqueeze(1).unsqueeze(1)
        
        # 加权融合
        return gate * h_feat + (1 - gate) * v_feat


class MambaSubBlock(nn.Module):
    """
    Mamba SubBlock (Independent Version):
    包含双向 SS2D 扫描机制的独立模块。
    """
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        mamba_cfg = dict(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        
        # 初始化 Mamba 层
        self.mamba_h = Mamba(**mamba_cfg)
        self.mamba_v = Mamba(**mamba_cfg)
        
        # 融合与 MLP
        self.fusion_gate = AdaptiveFusionGate(dim=dim, reduction=4)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TokenSpaceMLP(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        """
        Args:
            x: Input tensor with shape [B, C, H, W]
        Returns:
            Output tensor with shape [B, C, H, W]
        """
        B, C, H, W = x.shape
        L = H * W
        
        # 1. 转换到 Token 空间 [B, C, H, W] -> [B, H, W, C] -> [B, L, C]
        x_hw = x.permute(0, 2, 3, 1).contiguous() 
        x_token = x_hw.view(B, -1, C)
        
        # 2. Mamba SS2D 处理
        x_norm = self.norm1(x_token)
        x_hw_norm = x_norm.view(B, H, W, C)
        
        # --- 水平扫描 (Horizontal) ---
        x_h_flat = x_hw_norm.view(B, -1, C) 
        out_h_fwd = self.mamba_h(x_h_flat)
        out_h_bwd = self.mamba_h(x_h_flat.flip([1])).flip([1])
        out_h_combined = (out_h_fwd + out_h_bwd).view(B, H, W, C)
        
        # --- 垂直扫描 (Vertical) ---
        x_v_flat = x_hw_norm.transpose(1, 2).contiguous().view(B, -1, C)
        out_v_fwd = self.mamba_v(x_v_flat)
        out_v_bwd = self.mamba_v(x_v_flat.flip([1])).flip([1])
        out_v_combined = (out_v_fwd + out_v_bwd).view(B, W, H, C).transpose(1, 2)
        
        # --- 门控融合 ---
        mamba_out_hw = self.fusion_gate(out_h_combined, out_v_combined)
        mamba_out = mamba_out_hw.view(B, L, C)
        
        # Residual Connection 1
        x_token = x_token + self.drop_path(mamba_out)
        
        # 3. MLP 处理
        x_token_norm = self.norm2(x_token)
        mlp_out = self.mlp(x_token_norm)
        
        # Residual Connection 2
        x_token = x_token + self.drop_path(mlp_out)
        
        # 4. 还原回 Image 空间 [B, H, W, C] -> [B, C, H, W]
        x_out = x_token.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        return x_out