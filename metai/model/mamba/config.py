# metai/model/mamba/config.py

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Tuple, Literal, Union, List

class Config(BaseModel):
    """
    模型配置类
    适配 10帧 -> 20帧 预测任务，以及 30 通道输入 (含周期性 GIS 编码)。
    """

    # 1. 基础配置
    model_name: str = Field(default="met_mamba", description="模型名称")
    data_path: str = Field(default="data/samples.jsonl", description="数据索引文件路径 (.jsonl)")
    save_dir: str = Field(default="./output/mamba", description="训练输出目录")
    
    in_shape: Tuple[int, int, int, int] = Field(default=(10, 54, 256, 256), description="输入形状 (T, C, H, W)")
    out_seq_length: int = Field(default=20, description="输出序列长度 (预测帧数)")
    max_epochs: int = Field(default=100, description="最大训练轮数")

    # 2. 数据加载器配置
    batch_size: int = Field(default=4, description="批大小 (单卡)")
    seed: int = Field(default=42, description="全局随机种子")
    num_workers: int = Field(default=4, description="DataLoader 工作线程数")
    train_split: float = Field(default=0.8, description="训练集比例")
    val_split: float = Field(default=0.1, description="验证集比例")
    test_split: float = Field(default=0.1, description="测试集比例")

    # 3. Trainer 配置
    precision: Literal["16-mixed", "32", "64", "16-true", "bf16-mixed", "bf16-true", "32-true"] = Field(default="16-mixed", description="训练精度")
    accelerator: Literal["auto", "cpu", "cuda"] = Field(default="auto", description="加速器类型")
    devices: Union[int, str, List[int]] = Field(default="auto", description="设备编号")
    log_every_n_steps: int = Field(default=100, description="日志记录频率")
    val_check_interval: float = Field(default=1.0, description="验证频率")
    gradient_clip_val: float = Field(default=0.5, description="梯度裁剪阈值")
    gradient_clip_algorithm: Literal["norm", "value"] = Field(default="norm", description="梯度裁剪算法")
    enable_progress_bar: bool = Field(default=True, description="显示进度条")
    enable_model_summary: bool = Field(default=True, description="显示模型摘要")
    accumulate_grad_batches: int = Field(default=1, description="梯度累积步数")
    num_sanity_val_steps: int = Field(default=2, description="训练前健全性检查步数")

    # 4. 模型结构参数
    hid_S: int = Field(default=128, description="空间编码器隐藏层通道数")
    hid_T: int = Field(default=512, description="时序转换器隐藏层通道数")
    N_S: int = Field(default=4, description="空间编码器层数")
    N_T: int = Field(default=12, description="时序转换器层数")
    model_type: str = Field(default='mamba', description="时序模块类型")
    mlp_ratio: float = Field(default=4.0, description="MLP 扩展比例")
    drop: float = Field(default=0.0, description="Dropout 比率")
    drop_path: float = Field(default=0.1, description="Drop Path 比率")
    spatio_kernel_enc: int = Field(default=5, description="编码器卷积核大小")
    spatio_kernel_dec: int = Field(default=5, description="解码器卷积核大小")
    out_channels: int = Field(default=1, description="输出通道数")

    # 5. 损失函数配置
    loss_weight_l1: float = Field(default=1.0, description="L1 Loss 权重")
    loss_weight_ssim: float = Field(default=0.5, description="MS-SSIM 权重")
    loss_weight_csi: float = Field(default=1.0, description="Soft-CSI 权重")
    loss_weight_spectral: float = Field(default=0.05, description="Spectral 权重")
    loss_weight_evo: float = Field(default=0.2, description="Evolution 权重")
    
    # 6. 课程学习配置
    use_curriculum_learning: bool = Field(default=True, description="是否启用课程学习")

    # 7. 早停 (Early Stopping)
    early_stop_monitor: str = Field(default="val_score", description="监控指标")
    early_stop_mode: str = Field(default="max", description="早停模式")
    early_stop_min_delta: float = Field(default=1e-4, description="最小改善阈值")
    early_stop_patience: int = Field(default=50, description="容忍 Epoch 数 (加长以适应 Phase 3)")

    # 8. 优化器与调度器
    opt: str = Field(default="adamw", description="优化器")
    lr: float = Field(default=1e-4, description="初始学习率")
    weight_decay: float = Field(default=1e-2, description="权重衰减")
    filter_bias_and_bn: bool = Field(default=True, description="是否对 Bias 和 BN 层免除权重衰减")
    momentum: float = Field(default=0.9, description="SGD 动量")
    sched: str = Field(default="cosine", description="调度器")
    min_lr: float = Field(default=1e-5, description="最小学习率 (提高以支持 Phase 3)")
    warmup_lr: float = Field(default=1e-5, description="Warmup 初始学习率")
    warmup_epoch: int = Field(default=5, description="Warmup Epoch 数")
    decay_epoch: int = Field(default=30, description="Step Decay 的间隔")
    decay_rate: float = Field(default=0.1, description="Step Decay 的衰减率")

    @property
    def in_seq_length(self) -> int:
        return self.in_shape[0]

    @property
    def channels(self) -> int:
        return self.in_shape[1]

    @property
    def resize_shape(self) -> Tuple[int, int]:
        return (self.in_shape[2], self.in_shape[3])

    def to_dict(self) -> dict:
        data = self.model_dump()
        data['in_seq_length'] = self.in_seq_length
        data['channels'] = self.channels
        data['resize_shape'] = self.resize_shape
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Config':
        return cls(**data)

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())