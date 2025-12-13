# metai/model/mamba/config.py

from pydantic import BaseModel, Field, ConfigDict
from typing import Tuple, Union, List, Optional

class ModelConfig(BaseModel):
    """
    MetMamba 模型配置类
    
    用于定义模型的输入输出形状、结构超参数、训练参数及优化目标。
    """

    # =========================================================================
    # 1. 基础环境与路径配置 (Basic Environment & Paths)
    # =========================================================================
    model_name: str = Field(
        default="met_mamba", 
        description="模型实验名称，用于日志和Checkpoint命名"
    )
    data_path: str = Field(
        default="data/samples.jsonl", 
        description="数据索引文件路径 (.jsonl 格式)"
    )
    save_dir: str = Field(
        default="./output/mamba", 
        description="训练日志和模型权重的输出目录"
    )
    
    # =========================================================================
    # 2. 输入输出形状配置 (Input/Output Shapes)
    # =========================================================================
    in_shape: Tuple[int, int, int, int] = Field(
        default=(10, 54, 256, 256), 
        description="输入张量形状 (T, C, H, W)。T=历史帧数, C=通道数(含雷达+GIS)"
    )
    out_seq_length: int = Field(
        default=20, 
        description="预测序列长度 (输出帧数)"
    )
    out_channels: int = Field(
        default=1, 
        description="输出通道数 (通常为1，即雷达回波强度)"
    )

    # =========================================================================
    # 3. 数据加载配置 (Dataloader)
    # =========================================================================
    batch_size: int = Field(
        default=4, 
        description="单卡批大小 (Batch Size per GPU)"
    )
    num_workers: int = Field(
        default=4, 
        description="DataLoader 数据加载线程数"
    )
    seed: int = Field(
        default=42, 
        description="全局随机种子，保证实验可复现性"
    )
    
    # =========================================================================
    # 4. 模型结构超参数 (Model Architecture)
    # =========================================================================
    # 4.1 空间编码器/解码器 (Encoder/Decoder)
    hid_S: int = Field(
        default=64, 
        description="空间特征通道数 (Spatial Hidden Dim)"
    )
    N_S: int = Field(
        default=4, 
        description="空间编码器/解码器的层数 (Encoder/Decoder Layers)"
    )
    spatio_kernel_enc: int = Field(
        default=3, 
        description="编码器卷积核大小"
    )
    spatio_kernel_dec: int = Field(
        default=3, 
        description="解码器卷积核大小"
    )
    
    # 4.2 时序演变模块 (Temporal Translator - Mamba)
    hid_T: int = Field(
        default=256, 
        description="Mamba 中间层通道数 (Temporal Hidden Dim)"
    )
    N_T: int = Field(
        default=8, 
        description="Mamba 堆叠层数 (MidNet Layers)"
    )
    mlp_ratio: float = Field(
        default=4.0, 
        description="MLP 扩展比例 (Expansion Ratio)"
    )
    drop: float = Field(
        default=0.0, 
        description="Dropout 比率"
    )
    drop_path: float = Field(
        default=0.0, 
        description="Drop Path (Stochastic Depth) 比率"
    )

    # =========================================================================
    # 5. 训练与优化配置 (Training & Optimization)
    # =========================================================================
    max_epochs: int = Field(
        default=100, 
        description="最大训练轮数"
    )
    opt: str = Field(
        default="adamw", 
        description="优化器类型 (如: adamw, sgd)"
    )
    sched: str = Field(
        default="cosine", 
        description="学习率调度器类型 (如: cosine, one_cycle)"
    )
    lr: float = Field(
        default=1e-3, 
        description="初始学习率"
    )
    min_lr: float = Field(
        default=1e-5, 
        description="最小学习率 (Cosine Annealing 下界)"
    )
    warmup_epoch: int = Field(
        default=5, 
        description="预热轮数 (Warmup Epochs)"
    )
    weight_decay: float = Field(
        default=1e-2, 
        description="权重衰减 (L2 正则化系数)"
    )

    # =========================================================================
    # 6. 损失函数权重 (Loss Weights)
    # =========================================================================
    loss_weight_l1: float = Field(
        default=1.0, 
        description="L1 Loss (MAE) 权重"
    )
    loss_weight_ssim: float = Field(
        default=0.5, 
        description="MS-SSIM 结构相似性损失权重"
    )
    loss_weight_csi: float = Field(
        default=1.0, 
        description="Soft-CSI (临界成功指数) 损失权重"
    )
    loss_weight_spectral: float = Field(
        default=0.1, 
        description="频域距离损失权重 (抗模糊)"
    )
    loss_weight_evo: float = Field(
        default=0.5, 
        description="时序演变一致性损失权重"
    )
    use_curriculum_learning: bool = Field(
        default=True, 
        description="是否启用课程学习 (动态调整 Loss 权重)"
    )

    # =========================================================================
    # 7. 系统与训练器杂项 (System & Trainer Misc)
    # =========================================================================
    precision: str = Field(
        default="16-mixed", 
        description="训练精度 (16-mixed, 32)"
    )
    accelerator: str = Field(
        default="auto", 
        description="硬件加速器 (auto, gpu, cpu)"
    )
    devices: Union[int, str, List[int]] = Field(
        default="auto", 
        description="使用的设备编号"
    )
    check_val_every_n_epoch: int = Field(
        default=1, 
        description="每多少个 Epoch 执行一次验证"
    )
    
    # 早停配置 (Early Stopping)
    early_stop_monitor: str = Field(default="val_score", description="早停监控指标")
    early_stop_mode: str = Field(default="max", description="早停模式 (min/max)")
    early_stop_patience: int = Field(default=20, description="早停忍耐轮数")

    # =========================================================================
    # 8. 辅助属性与方法 (Properties & Methods)
    # =========================================================================
    
    @property
    def in_seq_length(self) -> int:
        """获取输入序列长度 (T)"""
        return self.in_shape[0]

    @property
    def channels(self) -> int:
        """获取输入通道数 (C)"""
        return self.in_shape[1]
    
    @property
    def resize_shape(self) -> Tuple[int, int]:
        """获取目标图像分辨率 (H, W)"""
        return (self.in_shape[2], self.in_shape[3])

    def to_dict(self) -> dict:
        """转换为字典格式，并补充推导属性"""
        data = self.model_dump()
        data['in_seq_length'] = self.in_seq_length
        data['channels'] = self.channels
        data['resize_shape'] = self.resize_shape
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ModelConfig':
        """从字典创建配置实例"""
        return cls(**data)

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())