# metai/dataset/met_dataloader_scwds.py

import os
import json
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from metai.dataset import MetSample
from metai.utils.met_config import get_config

class ScwdsDataset(Dataset):
    """
    SCWDS (Severe Convective Weather Dataset) 自定义数据集类。
    
    该类实现了 PyTorch Dataset 接口，负责从文件系统中读取气象样本数据。
    支持两种加载模式：
    1. **快速加载 (NPZ)**: 优先尝试加载预处理好的 `.npz` 文件（包含 input/target 张量），以提高 I/O 效率。
    2. **实时加载 (On-the-fly)**: 若 `.npz` 不存在，则使用 `MetSample` 类从原始数据源（雷达/NWP/标签文件）实时构建样本。
    """
    
    def __init__(self, data_path: str, is_train: bool = True, test_set: str = "TestSetB"):
        """
        初始化 SCWDS 数据集。

        Args:
            data_path (str): 样本索引文件路径 (.jsonl 格式)，每行包含一个样本的元数据字典。
            is_train (bool, optional): 模式标记。
                - True: 训练/验证模式，会加载标签数据 (Target)。
                - False: 推理模式，仅加载输入数据。
                默认为 True。
            test_set (str, optional): 测试集子目录名称 (如 "TestSetB")，用于构建原始文件路径。
        """
        self.data_path = data_path
        self.config = get_config()

        # 加载所有样本元数据
        self.samples = self._load_samples_from_jsonl(data_path)
        self.is_train = is_train
        self.test_set = test_set

        # 预处理数据缓存目录，用于加速训练 I/O
        self.npz_dir = "/data/zjobs/SevereWeather_AI_2025/CP/Train"
        
    def __len__(self) -> int:
        """
        返回数据集中的样本总数。

        Returns:
            int: 样本数量。
        """
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        获取指定索引的样本数据。

        优先读取预处理的 `.npz` 文件；若失败则回退到 `MetSample` 实时读取。

        Args:
            idx (int): 样本索引。
            
        Returns:
            tuple:包含以下元素的元组 (metadata, input_data, target_data, input_mask, target_mask):
                - metadata (None | Dict): 样本元数据（NPZ 模式下通常为 None，实时模式下为 Dict）。
                - input_data (np.ndarray): 输入序列张量。Shape: [T_in, C, H, W]。
                - target_data (np.ndarray | None): 目标序列张量 (如降水)。Shape: [T_out, 1, H, W]。推理模式下为 None。
                - input_mask (None | np.ndarray): 输入数据的有效性掩码 (NPZ 模式下暂未存储，返回 None)。
                - target_mask (np.ndarray | None): 目标数据的有效性掩码。Shape: [T_out, 1, H, W]。
        """
        record = self.samples[idx]
        sample_id = record.get("sample_id")
        timestamps = record.get("timestamps")

        # 仅在训练模式下尝试读取缓存的 NPZ 文件
        if self.is_train:
            npz_path = os.path.join(self.npz_dir, f"{sample_id}.npz")

            if os.path.exists(npz_path):
                try:
                    # 使用 allow_pickle=True 以确保兼容性，读取压缩的 numpy 数组
                    with np.load(npz_path, allow_pickle=True) as data:
                        input_data = data['input_data']   # Shape: [T_in, C, 256, 256]
                        target_data = data['target_data'] # Shape: [T_out, 1, 256, 256]
                        target_mask = data['target_mask'] # Shape: [T_out, 1, 256, 256]

                    # 返回与 MetSample.to_numpy() 接口一致的五元组结构
                    # 注意：NPZ 缓存中暂未包含 metadata 和 input_mask，故返回 None
                    return None, input_data, target_data, None, target_mask

                except Exception as e:
                    # 如果 NPZ 读取损坏，打印警告并回退到原始加载方式，保证训练鲁棒性
                    print(f"[WARNING] NPZ load failed for {sample_id}: {e}.")
        
        # 回退机制：创建 MetSample 实例来处理原始文件的读取、归一化和预处理
        sample = MetSample.create(
            sample_id,
            timestamps,
            config=self.config,
            is_train=self.is_train,
            test_set=self.test_set
        )
        
        # 返回: metadata, input_data, target_data, input_mask, target_mask
        return sample.to_numpy() 
                        
    def _load_samples_from_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """
        解析 JSONL 索引文件。

        Args:
            file_path (str): JSONL 文件绝对路径。

        Returns:
            List[Dict[str, Any]]: 包含样本信息的字典列表。
        """
        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    samples.append(sample)
        return samples

class ScwdsDataModule(LightningDataModule):
    """
    基于 PyTorch Lightning 的数据模块 (DataModule)。
    
    统一管理 SCWDS 数据的生命周期，包括：
    1. **数据准备**: 解析索引文件。
    2. **数据集划分**: 自动将数据集划分为训练集 (Train)、验证集 (Val) 和测试集 (Test)。
    3. **加载器构建**: 提供标准化的 DataLoader，支持多进程加载和自定义 Collate 函数。
    """

    def __init__(
        self,
        data_path: str = "data/samples.jsonl",
        batch_size: int = 4,
        num_workers: int = 8,
        pin_memory: bool = True,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
    ):
        """
        初始化 DataModule。

        Args:
            data_path (str): 样本索引文件路径。
            batch_size (int): 每个批次的样本数量 (Batch Size)。
            num_workers (int): DataLoader 的子进程数量。
            pin_memory (bool): 是否将数据固定在 CUDA 锁页内存中 (Pin Memory)，加速 Host-to-Device 传输。
            train_split (float): 训练集划分比例 (0.0 ~ 1.0)。
            val_split (float): 验证集划分比例。
            test_split (float): 测试集划分比例。
            seed (int): 随机种子，用于固定数据集划分结果 (random_split)。
        """
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.pin_memory = pin_memory
        self.seed = seed

    def setup(self, stage: Optional[str] = None):
        """
        执行数据集的构建与划分逻辑。
        
        根据传入的 `stage` 参数，分别准备 fit (训练/验证), test (测试) 或 infer (推理) 阶段的数据。
        
        Args:
            stage (str, optional): 阶段标识。可选值: 'fit', 'validate', 'test', 'infer'。
        """
        # --- 推理模式 (Inference) ---
        if stage == "infer":
            # 推理模式下，直接使用全量数据，无需分割，且 is_train=False
            self.infer_dataset = ScwdsDataset(
                self.data_path, 
                is_train=False
            )
            print(f"[INFO] Infer dataset: {self.infer_dataset.test_set}, size = {len(self.infer_dataset)}")
            return
        
        # --- 训练/验证/测试模式 (Fit/Test) ---
        if not hasattr(self, 'dataset'):
            self.dataset = ScwdsDataset(
                data_path=self.data_path,
                is_train=True
            )
            
            total_size = len(self.dataset)
            
            # 异常处理：如果数据集为空
            if total_size == 0:
                print("[WARNING] Dataset is empty, skipping split")
                return
            
            # 根据比例计算各子集大小
            train_size = int(self.train_split * total_size)
            val_size = int(self.val_split * total_size)
            test_size = total_size - train_size - val_size
            
            # 边界情况处理：确保训练集至少有一个样本，防止 DataLoader 报错
            if train_size == 0 and total_size > 0:
                train_size = 1
                test_size = total_size - train_size - val_size
            
            lengths = [train_size, val_size, test_size]

            # 创建确定性随机生成器，确保每次运行划分结果一致
            generator = torch.Generator().manual_seed(self.seed)

            # 执行随机划分
            self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
                self.dataset, lengths, generator=generator
            )
            
            print(f"[INFO] Dataset split: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}, Test={len(self.test_dataset)}")

    def _collate_fn(self, batch):
        """
        训练/验证/测试阶段的 Collate 函数。

        将 DataLoader 取出的样本列表堆叠为 Batch 张量。

        Args:
            batch (List[tuple]): 由 __getitem__ 返回的样本列表。

        Returns:
            tuple:
                - metadata (None): 元数据 (暂未在 Batch 中保留)。
                - input_batch (Tensor): 输入数据。Shape: [B, T_in, C, H, W]。
                - target_batch (Tensor): 目标数据。Shape: [B, T_out, 1, H, W]。
                - input_mask (None): 输入掩码 (暂未处理)。
                - target_mask_batch (Tensor): 目标掩码。Shape: [B, T_out, 1, H, W]。
        """
        # metadata_batch = []
        input_tensors = []
        target_tensors = []
        target_mask_tensors = []
        # input_mask_tensors = []

        # 解包 batch 中的每个样本
        for _, input_np, target_np, _, target_mask_np in batch:
            # metadata_batch.append(metadata)
            input_tensors.append(torch.from_numpy(input_np).float())
            target_tensors.append(torch.from_numpy(target_np).float())
            target_mask_tensors.append(torch.from_numpy(target_mask_np).bool())
            # input_mask_tensors.append(torch.from_numpy(input_mask_np).bool())

        # 使用 torch.stack 将列表转换为张量，并在维度 0 (Batch) 上堆叠
        # .contiguous() 确保内存连续，优化后续计算效率
        input_batch = torch.stack(input_tensors, dim=0).contiguous()
        target_batch = torch.stack(target_tensors, dim=0).contiguous()
        target_mask_batch = torch.stack(target_mask_tensors, dim=0).contiguous()
        # input_mask_batch = torch.stack(input_mask_tensors, dim=0).contiguous()
        
        return None, input_batch, target_batch, None, target_mask_batch

    def _collate_fn_infer(self, batch):
        """
        推理阶段的 Collate 函数。

        与 `_collate_fn` 的区别在于不处理 Target 数据，且保留 Metadata 以便于后续结果保存。

        Args:
            batch (List[tuple]): 样本列表。

        Returns:
            tuple:
                - metadata_batch (List[Dict]): 样本元数据列表。
                - input_batch (Tensor): 输入数据。Shape: [B, T_in, C, H, W]。
                - input_mask_batch (Tensor): 输入掩码。Shape: [B, T_in, C, H, W]。
        """
        metadata_batch = []
        input_tensors = []
        input_mask_tensors = []

        for metadata, input_np, _, input_mask_np, _ in batch:
            metadata_batch.append(metadata)
            input_tensors.append(torch.from_numpy(input_np).float())
            input_mask_tensors.append(torch.from_numpy(input_mask_np).bool())
        
        input_batch = torch.stack(input_tensors, dim=0).contiguous()
        input_mask_batch = torch.stack(input_mask_tensors, dim=0).contiguous()
        
        return metadata_batch, input_batch, input_mask_batch


    def train_dataloader(self):
        """
        获取训练集 DataLoader。

        Returns:
            DataLoader: 训练数据加载器 (shuffle=True)。
        """
        if not hasattr(self, 'train_dataset'):
            self.setup('fit')
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """
        获取验证集 DataLoader。

        Returns:
            DataLoader: 验证数据加载器 (shuffle=False)。
        """
        if not hasattr(self, 'val_dataset'):
            self.setup('fit')
            
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False, # 验证集不需要打乱
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """
        获取测试集 DataLoader。

        Returns:
            DataLoader: 测试数据加载器。
        """
        if not hasattr(self, 'test_dataset'):
            self.setup('test')
            
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self._collate_fn,
        )

    def infer_dataloader(self) -> Optional[DataLoader]:
        """
        获取推理集 DataLoader。

        Returns:
            DataLoader: 推理数据加载器 (使用 _collate_fn_infer)。
        """
        if not hasattr(self, 'infer_dataset'):
            self.setup('infer')
            
        return DataLoader(
            self.infer_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self._collate_fn_infer 
        )