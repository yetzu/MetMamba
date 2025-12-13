# metai/model/mamba/module.py

import torch
import torch.nn as nn
from timm.layers.drop import DropPath
from timm.layers.weight_init import trunc_normal_

from mamba_ssm import Mamba

class TokenSpaceMLP(nn.Module):
    """
    Token Space MLP (多层感知机)
    
    作用：
        在 Token 维度上进行特征变换，替代了传统 CNN 中的 1x1 卷积或 MixMLP。
        专为 [B, L, C] 或 [B, H, W, C] 格式的张量设计。
    
    参数:
        in_features (int): 输入特征维度
        hidden_features (int): 隐藏层维度 (默认等于 in_features)
        out_features (int): 输出特征维度 (默认等于 in_features)
        act_layer (nn.Module): 激活函数 (默认 nn.GELU)
        drop (float): Dropout 比率
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
        """
        Args:
            x: Input tensor, shape [B, L, C] or [B, H, W, C]
        Returns:
            Output tensor, same shape as input
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AdaptiveFusionGate(nn.Module):
    """
    自适应门控融合模块 (Adaptive Fusion Gate)
    
    作用：
        用于动态融合 SS2D 机制中产生的“水平扫描特征”和“垂直扫描特征”。
        通过全局上下文信息计算门控权重，实现 soft-selection。
    
    参数:
        dim (int): 特征通道数
        reduction (int): 门控网络中间层的缩放比例
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
            h_feat: 水平扫描特征, shape [B, H, W, C]
            v_feat: 垂直扫描特征, shape [B, H, W, C]
        Returns:
            融合后的特征, shape [B, H, W, C]
        """
        # 1. 提取全局上下文: (h + v) / 2 -> Global Avg Pool -> [B, C]
        combined_context = (h_feat + v_feat) / 2.0
        context = combined_context.mean(dim=(1, 2)) 
        
        # 2. 生成门控权重: [B, C] -> [B, 1, 1, C]
        gate = self.gate_net(context).unsqueeze(1).unsqueeze(1)
        
        # 3. 加权融合
        return gate * h_feat + (1 - gate) * v_feat


class MambaSubBlock(nn.Module):
    """
    Mamba SubBlock (SS2D 核心模块)
    
    架构设计：
        包含双向 SS2D (Spatial-Temporal 2D Scanning) 机制的 Mamba 模块。
        通过水平和垂直两个方向的扫描，捕捉 2D 空间特征的长程依赖。
    
    流程:
        Input -> Norm -> SS2D (Horizontal & Vertical Mamba) -> Gate Fusion -> Residual -> Norm -> MLP -> Residual -> Output
    
    参数:
        dim (int): 输入通道数
        mlp_ratio (float): MLP 隐藏层扩展比例
        drop (float): Dropout 比率
        drop_path (float): Stochastic Depth (Drop Path) 比率
        act_layer (nn.Module): 激活函数
    """
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        # Mamba 配置
        # d_state: SSM 状态维度 (N)
        # d_conv: 局部卷积核大小
        # expand: 输入投影扩展倍数
        mamba_cfg = dict(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        
        self.mamba_h = Mamba(**mamba_cfg) # 水平方向
        self.mamba_v = Mamba(**mamba_cfg) # 垂直方向
        
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
            x: Input tensor, shape [B, C, H, W]  (注意：输入是 Image 格式)
        Returns:
            Output tensor, shape [B, C, H, W]
        """
        B, C, H, W = x.shape
        L = H * W
        
        # ==========================================================
        # 1. 预处理: 转换到 Token 空间 [B, C, H, W] -> [B, H, W, C]
        # ==========================================================
        x_hw = x.permute(0, 2, 3, 1).contiguous() 
        x_token = x_hw.view(B, -1, C) # [B, L, C] 用于 Residual
        
        # Norm
        x_norm = self.norm1(x_token)
        x_hw_norm = x_norm.view(B, H, W, C)
        
        # ==========================================================
        # 2. SS2D Mamba 扫描 (Spatial-Temporal 2D Scanning)
        # ==========================================================
        
        # --- A. 水平扫描 (Horizontal Scan) ---
        # 展平 -> [B, L, C]
        x_h_flat = x_hw_norm.view(B, -1, C) 
        
        # Forward + Backward Mamba
        # 翻转逻辑: flip([1]) 在序列维度 L 上翻转
        out_h_fwd = self.mamba_h(x_h_flat)
        out_h_bwd = self.mamba_h(x_h_flat.flip([1])).flip([1])
        out_h_combined = (out_h_fwd + out_h_bwd).view(B, H, W, C)
        
        # --- B. 垂直扫描 (Vertical Scan) ---
        # 转置 H, W -> [B, W, H, C] -> 展平 [B, L, C]
        # 这样 Mamba 沿着原来的 W 轴扫描，实际上是图像的列方向
        x_v_flat = x_hw_norm.transpose(1, 2).contiguous().view(B, -1, C)
        
        # Forward + Backward Mamba
        out_v_fwd = self.mamba_v(x_v_flat)
        out_v_bwd = self.mamba_v(x_v_flat.flip([1])).flip([1])
        
        # 还原形状: [B, L, C] -> [B, W, H, C] -> Transpose回 [B, H, W, C]
        out_v_combined = (out_v_fwd + out_v_bwd).view(B, W, H, C).transpose(1, 2)
        
        # --- C. 门控融合 (Adaptive Gating) ---
        # 融合水平和垂直特征 [B, H, W, C]
        mamba_out_hw = self.fusion_gate(out_h_combined, out_v_combined)
        mamba_out = mamba_out_hw.view(B, L, C)
        
        # Residual Connection 1
        x_token = x_token + self.drop_path(mamba_out)
        
        # ==========================================================
        # 3. MLP 处理 (Feed Forward)
        # ==========================================================
        x_token_norm = self.norm2(x_token)
        mlp_out = self.mlp(x_token_norm)
        
        # Residual Connection 2
        x_token = x_token + self.drop_path(mlp_out)
        
        # ==========================================================
        # 4. 后处理: 还原回 Image 空间 [B, C, H, W]
        # ==========================================================
        x_out = x_token.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        return x_out