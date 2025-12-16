# metai/dataset/met_sample.py

import os
import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Optional, Dict, Union, Tuple
from datetime import datetime, timedelta
import numpy as np
import cv2
from metai.utils import get_config
from metai.utils import MetConfig, MetLabel, MetRadar, MetNwp, MetGis, MetVar

# 默认通道配置列表
_DEFAULT_CHANNELS: List[Union[MetLabel, MetRadar, MetNwp, MetGis]] = [
    MetLabel.RA, MetRadar.CR, MetRadar.CAP20, MetRadar.CAP30, MetRadar.CAP40, MetRadar.CAP50, MetRadar.CAP60, MetRadar.CAP70, MetRadar.ET, MetRadar.HBR, MetRadar.VIL,
    MetNwp.WS925, MetNwp.WS700, MetNwp.WS500, MetNwp.Q1000, MetNwp.Q850, MetNwp.Q700, MetNwp.PWAT, MetNwp.PE, MetNwp.TdSfc850, MetNwp.LCL, MetNwp.KI, MetNwp.CAPE,
    MetGis.LAT, MetGis.LON, MetGis.DEM, MetGis.MONTH_SIN, MetGis.MONTH_COS, MetGis.HOUR_SIN, MetGis.HOUR_COS
]


@dataclass
class MetSample:
    """
    气象样本数据加载器与容器 (Sample Loader)。

    该类负责管理和加载单个训练/预测样本所需的所有数据。核心功能包括：
    1. 元数据解析：从样本 ID 中提取任务、区域、时间和站点信息。
    2. 多源数据融合：加载雷达观测、数值预报 (NWP)、地理信息 (GIS) 和标签数据。
    3. 数据预处理：执行缺失值填充、异常值处理、Min-Max 归一化和统一尺寸调整 (Resize)。
    4. 时序构建：将历史观测序列和未来预报序列构建为模型可用的张量格式。

    Attributes:
        sample_id (str): 样本唯一标识符 (例如: 'CP_AH_202405111200_58238_SA_0')。
        timestamps (List[str]): 该样本涉及的所有时间戳列表 (包含历史 T_in 和未来 T_out)。
        met_config (MetConfig): 全局配置对象，提供路径和格式定义。
        is_train (bool): 模式标记。True 表示训练模式 (加载 Label)，False 表示推理模式。
        test_set (str): 测试集目录名称 (如 "TestSetB")。
        in_seq_length (int): 输入历史序列长度 T_in (默认为 10)。
        out_seq_length (int): 输出预测序列长度 T_out (默认为 20)。
        channels (List): 需要加载的通道列表 (MetRadar, MetNwp, MetGis 等枚举成员)。
        channel_size (int): 通道总数。
        default_shape (Tuple[int, int]): 目标空间分辨率 (H, W)，默认为 (301, 301)。
    """
    sample_id: str
    timestamps: List[str]
    met_config: MetConfig
    is_train: bool = field(default_factory=lambda: True)
    test_set: str = field(default_factory=lambda: "TestSetB")
    
    in_seq_length: int = field(default_factory=lambda: 10)
    out_seq_length: int = field(default_factory=lambda: 20)
    
    channels: List[Union[MetLabel, MetRadar, MetNwp, MetGis]] = field(
        default_factory=lambda: _DEFAULT_CHANNELS.copy()
    )
    channel_size: int = field(default_factory=lambda: len(_DEFAULT_CHANNELS))
    
    default_shape: Tuple[int, int] = field(default_factory=lambda: (301, 301))

    # 内部缓存，用于存储静态 GIS 数据，避免重复 IO
    _gis_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict, init=False, repr=False)
    # 缓存解析后的 ID 片段
    _sample_id_parts: Optional[List[str]] = field(default=None, init=False, repr=False)

    @classmethod
    def create(cls, sample_id: str, timestamps: List[str], config: Optional['MetConfig'] = None, **kwargs) -> 'MetSample':
        """
        工厂方法：创建 MetSample 实例。
        
        Args:
            sample_id (str): 样本 ID。
            timestamps (List[str]): 时间戳列表。
            config (Optional[MetConfig]): 配置对象。若为 None，则自动获取全局配置。
            **kwargs: 其他初始化参数。

        Returns:
            MetSample: 初始化后的样本对象。
        """
        if config is None:
            config = get_config()

        return cls(
            sample_id=sample_id,
            timestamps=timestamps,
            met_config=config,
            **kwargs
        )
    
    def _get_sample_id_parts(self) -> List[str]:
        """
        解析 sample_id。
        
        格式通常为：Task_Region_Time_Station_Radar_Batch
        
        Returns:
            List[str]: 分割后的字符串列表。
        """
        if self._sample_id_parts is None:
            self._sample_id_parts = self.sample_id.split('_')
        return self._sample_id_parts
    
    # --- 元数据属性 (Properties) ---
    
    @cached_property
    def task_id(self) -> str:
        """任务 ID (如 CP)。"""
        return self._get_sample_id_parts()[0]

    @cached_property
    def region_id(self) -> str:
        """区域 ID (如 AH)。"""
        return self._get_sample_id_parts()[1]

    @cached_property
    def time_id(self) -> str:
        """时间标识。"""
        return self._get_sample_id_parts()[2]

    @cached_property
    def station_id(self) -> str:
        """站点 ID (如 58238)。"""
        return self._get_sample_id_parts()[3]

    @cached_property
    def radar_type(self) -> str:
        """雷达类型 (如 SA)。"""
        return self._get_sample_id_parts()[4]

    @cached_property
    def batch_id(self) -> str:
        """批次 ID。"""
        return self._get_sample_id_parts()[5]

    @cached_property
    def case_id(self) -> str:
        """
        获取 Case ID。
        
        组合 Task, Region, Time, Station 作为 Case 目录名。
        
        Returns:
            str: Case ID (如 CP_AH_202405111200_58238)。
        """
        parts = self._get_sample_id_parts()
        return '_'.join(parts[:4])

    @cached_property
    def base_path(self) -> str:
        """
        构建样本数据的文件系统根路径。
        
        路径结构: Root/Task/DatasetType/Region/CaseID
        
        Returns:
            str: 绝对路径。
        """
        return os.path.join(
            self.met_config.root_path,
            self.task_id,
            "TrainSet" if self.is_train else self.test_set,
            self.region_id,
            self.case_id,
        )
    
    @property
    def root_path(self) -> str:
        return self.met_config.root_path
    
    @property
    def gis_data_path(self) -> str:
        return self.met_config.gis_data_path
    
    @property
    def file_date_format(self) -> str:
        return self.met_config.file_date_format
    
    @property
    def nwp_prefix(self) -> str:
        return self.met_config.nwp_prefix
    
    @property
    def metadata(self) -> Dict:
        """
        获取不包含缓存数据的元数据字典。
        
        Returns:
            Dict: 包含样本配置信息的字典。
        """
        metadata_dict = vars(self).copy()
        metadata_dict.pop('_gis_cache', None)
        return metadata_dict

    # --- 工具方法 (Utils) ---

    def str_to_datetime(self, time_str: str) -> datetime:
        """将字符串解析为 datetime 对象。"""
        return datetime.strptime(time_str, self.file_date_format)

    def datetime_to_str(self, datetime_obj: datetime) -> str:
        """将 datetime 对象格式化为字符串。"""
        return datetime_obj.strftime(self.file_date_format)

    def _ensure_shape(self, data: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        强制调整数据和掩码的尺寸以匹配 default_shape (H, W)。
        
        Args:
            data (np.ndarray): 输入数据数组。
            mask (np.ndarray): 输入掩码数组。

        Returns:
            Tuple[np.ndarray, np.ndarray]: 调整尺寸后的 (data, mask)。
                - data: 形状为 (H, W) 的 float32 数组。
                - mask: 形状为 (H, W) 的 bool 数组。
                如果输入无效或 Resize 失败，返回全零数组。
        """
        # 防御性检查：空数组或包含0维的数组
        if data is None or data.size == 0 or 0 in data.shape:
            return np.zeros(self.default_shape, dtype=np.float32), np.zeros(self.default_shape, dtype=bool)

        if data.shape != self.default_shape:
            # cv2.resize 接收参数为 (width, height)，对应 numpy 的 (col, row)
            # self.default_shape 是 (Height, Width) -> (301, 301)
            dsize = (self.default_shape[1], self.default_shape[0])
            try:
                # 线性插值用于连续数值数据
                data = cv2.resize(data, dsize, interpolation=cv2.INTER_LINEAR)
                # 最近邻插值用于布尔掩码（防止产生中间值）
                mask = cv2.resize(mask.astype(np.uint8), dsize, interpolation=cv2.INTER_NEAREST).astype(bool)
            except cv2.error:
                # Resize 失败回退
                return np.zeros(self.default_shape, dtype=np.float32), np.zeros(self.default_shape, dtype=bool)
        return data, mask

    def normalize(self, file_path: str, min_value: float = 0, max_value: float = 300) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        加载数据并进行 Min-Max 归一化。
        
        Args:
            file_path (str): 数据文件路径 (.npy)。
            min_value (float): 物理量最小值。
            max_value (float): 物理量最大值。

        Returns:
            Tuple: (归一化后的数据, 有效值掩码)。
                - data: [0, 1] 范围内的 float32 数组。
                - mask: 指示有效区域的 boolean 数组 (非 NaN/Inf)。
                如果加载失败返回 (None, None)。
        """
        try:
            data = np.load(file_path)
            if data.size == 0 or 0 in data.shape:
                return None, None
            
            # 记录非 NaN/Inf 的有效区域
            valid_mask = np.isfinite(data)
            # 填充 NaN/Inf 为边界值，防止计算溢出
            data = np.nan_to_num(data, nan=min_value, neginf=min_value, posinf=max_value)
            
            # 归一化计算：(x - min) / (max - min)
            inv_denom = 1.0 / (max_value - min_value)
            scale = (data - min_value) * inv_denom
            np.clip(scale, 0.0, 1.0, out=scale)
            
            return scale.astype(np.float32, copy=False), valid_mask.astype(bool, copy=False)
        except Exception:
            return None, None

    def normalize_with_mask(self, file_path: str, min_value: float = 0, max_value: float = 300, missing_value: float = -9) -> tuple[np.ndarray, np.ndarray]:
        """
        加载数据，排除特定缺失值并归一化。
        
        通常用于 Label 数据加载，需要明确处理业务定义的缺失值标记 (如 -9.0)。
        
        Args:
            file_path (str): 数据文件路径。
            min_value (float): 最小值。
            max_value (float): 最大值。
            missing_value (float): 缺失值标记。

        Returns:
            tuple[np.ndarray, np.ndarray]: (data, mask)。加载失败时返回全零数组。
        """
        try:
            data = np.load(file_path)
            if data.size == 0 or 0 in data.shape:
                default_data = np.zeros(self.default_shape, dtype=np.float32)
                default_mask = np.zeros(self.default_shape, dtype=bool)
                return default_data, default_mask

            # 构建掩码：排除缺失值且必须是有限数
            valid_mask = (data != missing_value) & np.isfinite(data)
            
            data = np.nan_to_num(data, nan=min_value, neginf=min_value, posinf=max_value)
            inv_denom = 1.0 / (max_value - min_value)
            scale = (data - min_value) * inv_denom
            np.clip(scale, 0.0, 1.0, out=scale)
            
            return scale.astype(np.float32, copy=False), valid_mask.astype(bool, copy=False)
        except Exception:
            default_data = np.zeros(self.default_shape, dtype=np.float32)
            default_mask = np.zeros(self.default_shape, dtype=bool)
            return default_data, default_mask
    
    # --- 核心数据加载逻辑 ---

    def load_input_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载模型输入张量。
        
        逻辑流程：
        1. 预加载 GIS 静态数据到缓存。
        2. 加载过去时间步 (in_seq_length) 的所有通道数据 (Radar, NWP, GIS)。
        3. 加载未来时间步 (out_seq_length) 的 NWP 数据 (如果配置了 NWP 通道且数据完整)。
        4. 特征融合：将未来的 NWP 数据“折叠 (Fold)”并拼接到输入时间步的通道维度上，
           使模型在输入阶段就能感知未来的数值预报信息。
        
        Returns:
            tuple:
                - input_data (np.ndarray): 输入数据张量。
                  形状为 [T_in, C_total, H, W]。
                  其中 C_total = C_base + (C_nwp * fold_factor)。
                - input_mask (np.ndarray): 对应的数据有效性掩码。
        """
        past_timesteps = self.timestamps[:self.in_seq_length] 
        future_start = self.in_seq_length
        future_end = future_start + self.out_seq_length
        future_timesteps = self.timestamps[future_start:future_end]
        
        self._preload_gis_data()

        # --- A. 加载基础输入序列 (Past Series) ---
        # 形状: [T_in, C_base, H, W]
        past_series = []
        past_masks = []
        
        for i, timestamp in enumerate(past_timesteps):
            frames = []
            masks = []
            for channel in self.channels:
                # 使用带 fallback 的加载器，如果当前时刻缺失，尝试前后时刻填充
                data, mask = self._load_channel_frame_with_fallback(channel, timestamp, i, self.timestamps)
                frames.append(data)
                masks.append(mask)
            # Stack channels: (C, H, W)
            past_series.append(np.stack(frames, axis=0))
            past_masks.append(np.stack(masks, axis=0))
            
        # Stack time: (T_in, C, H, W)
        input_base = np.stack(past_series, axis=0)
        mask_base = np.stack(past_masks, axis=0)

        # --- B. 加载未来 NWP 引导序列 (Future NWP) ---
        nwp_channels = [c for c in self.channels if isinstance(c, MetNwp)]
        
        # 仅当存在 NWP 通道且未来时间步完整时执行
        if len(nwp_channels) > 0 and len(future_timesteps) == self.out_seq_length:
            future_nwp_series = []
            future_nwp_masks = []
            
            for i, timestamp in enumerate(future_timesteps):
                frames = []
                masks = []
                for channel in nwp_channels:
                    global_idx = future_start + i
                    data, mask = self._load_channel_frame_with_fallback(channel, timestamp, global_idx, self.timestamps)
                    frames.append(data)
                    masks.append(mask)
                future_nwp_series.append(np.stack(frames, axis=0))
                future_nwp_masks.append(np.stack(masks, axis=0))
            
            # (T_out, C_nwp, H, W)
            input_future_nwp = np.stack(future_nwp_series, axis=0)
            mask_future_nwp = np.stack(future_nwp_masks, axis=0)
            
            # --- C. 数据折叠与拼接 ---
            # 将未来的 NWP 数据折叠到输入时间步的通道维度上。
            # 例如：输入 10 帧，输出 20 帧。fold_factor = 2。
            # 意味着每个输入时间步将附带 2 个未来时刻的 NWP 预报信息。
            B_t, C_n, H_t, W_t = input_future_nwp.shape
            fold_factor = self.out_seq_length // self.in_seq_length
            
            if self.out_seq_length % self.in_seq_length == 0:
                # Reshape: (T_in, fold_factor * C_nwp, H, W)
                input_folded = input_future_nwp.reshape(self.in_seq_length, fold_factor * C_n, H_t, W_t)
                mask_folded = mask_future_nwp.reshape(self.in_seq_length, fold_factor * C_n, H_t, W_t)
                # Concatenate along channel axis (axis=1)
                input_data = np.concatenate([input_base, input_folded], axis=1)
                input_mask = np.concatenate([mask_base, mask_folded], axis=1)
            else:
                # 如果无法整除折叠，则仅返回基础输入（保持对齐）
                input_data = input_base
                input_mask = mask_base
        else:
            input_data = input_base
            input_mask = mask_base

        return input_data, input_mask

    def load_target_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        加载预测目标数据 (Ground Truth)。
        
        通常是雷达或降水 Label (RA)。
        
        Returns:
            tuple:
                - target_data (np.ndarray): 目标数据张量 [T_out, 1, H, W]。
                - valid_mask (np.ndarray): 有效性掩码 [T_out, 1, H, W]。
        """
        target_data = []
        valid_mask = []

        start_idx = self.in_seq_length
        end_idx = start_idx + self.out_seq_length
        target_timestamps = self.timestamps[start_idx : end_idx]

        for timestamp in target_timestamps:
            # 目标固定为 MetLabel.RA (降水)
            file_path = os.path.join(
                self.base_path,
                MetVar.LABEL.name,
                MetLabel.RA.name,
                f"{self.task_id}_Label_{MetLabel.RA.name}_{self.station_id}_{timestamp}.npy"
            )

            min_val, max_val = self._get_channel_limits(MetLabel.RA)
            missing_val = getattr(MetLabel.RA, 'missing_value', -9.0)
            data, mask = self.normalize_with_mask(file_path, min_val, max_val, missing_value=missing_val)
            
            if data is not None and mask is not None:
                data, mask = self._ensure_shape(data, mask)

            target_data.append(data)
            valid_mask.append(mask)

        # 增加通道维度 axis=1 -> (T_out, 1, H, W)
        target_data = np.expand_dims(np.stack(target_data, axis=0), axis=1)
        valid_mask = np.expand_dims(np.stack(valid_mask, axis=0), axis=1)
        return target_data, valid_mask

    def to_numpy(self) -> Tuple[Dict, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        将样本转换为 Numpy 数组格式，供 PyTorch Dataset 使用。
        
        Returns:
            Tuple: (metadata, input_data, target_data, input_mask, target_mask)
        """
        input_data, input_mask = self.load_input_data()
        
        if self.is_train:
            target_data, target_mask = self.load_target_data()
            return self.metadata, input_data, target_data, input_mask, target_mask
        else:
            return self.metadata, input_data, None, input_mask, None
    
    def _get_channel_limits(self, channel: Union[MetLabel, MetRadar, MetNwp, MetGis]) -> tuple[float, float]:
        """获取通道配置的物理量极值 (min, max)，用于归一化。"""
        return (
            float(getattr(channel, "min", 0.0)),
            float(getattr(channel, "max", 1.0))
        )

    # --- 具体类型的帧加载器 (Frame Loaders) ---

    def _load_label_frame(self, var: MetLabel, timestamp: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """加载标签数据帧。"""
        file_path = os.path.join(
            self.base_path, var.parent, var.value, 
            f"{self.task_id}_Label_{var.value}_{self.station_id}_{timestamp}.npy"
        )
        min_val, max_val = self._get_channel_limits(var)
        data, mask = self.normalize(file_path, min_val, max_val)
        
        if data is not None and mask is not None:
            data, mask = self._ensure_shape(data, mask)
        return data, mask

    def _load_radar_frame(self, var: MetRadar, timestamp: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """加载雷达数据帧。"""
        file_path = os.path.join(
            self.base_path, var.parent, var.value, 
            f"{self.task_id}_RADA_{self.station_id}_{timestamp}_{self.radar_type}_{var.value}.npy"
        )
        min_val, max_val = self._get_channel_limits(var)
        data, mask = self.normalize(file_path, min_val, max_val)
        
        if data is not None and mask is not None:
            data, mask = self._ensure_shape(data, mask)
        return data, mask

    def _load_nwp_frame(self, var: MetNwp, timestamp: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]: # type: ignore
        """
        加载 NWP 数据。
        
        包含时间对齐逻辑：NWP 数据通常是逐小时的，而观测数据可能是分钟级的。
        策略：<30分向下取整，>=30分向上取整到最近的小时 (Nearest Hour)。
        """
        obs_time = self.str_to_datetime(timestamp)

        if obs_time.minute < 30:
            obs_time = obs_time.replace(minute=0)
        else:
            obs_time = obs_time.replace(minute=0) + timedelta(hours=1)

        file_path = os.path.join(
            self.base_path, var.parent, var.name, 
            f"{self.task_id}_{self.nwp_prefix}_{self.station_id}_{self.datetime_to_str(obs_time)}_{var.value}.npy"
        )
        min_val, max_val = self._get_channel_limits(var)
        data, mask = self.normalize(file_path, min_val, max_val)
        
        if data is None or mask is None:
            return None, None
            
        # NWP 数据分辨率可能与雷达不同，强制 Resize 到 default_shape
        dsize = (self.default_shape[1], self.default_shape[0])
        try:
            resized_data = cv2.resize(data, dsize, interpolation=cv2.INTER_LINEAR)
            resized_mask = cv2.resize(mask.astype(np.uint8), dsize, interpolation=cv2.INTER_NEAREST).astype(bool)
        except cv2.error:
            return None, None
        
        return resized_data.astype(np.float32, copy=False), resized_mask

    def _load_gis_frame(self, var: MetGis, timestamp: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        加载地理信息数据 (GIS)。
        
        包含两种类型：
        1. 动态生成的时间编码 (Sin/Cos of Month/Hour)：根据时间戳计算，无需读取文件。
        2. 静态 GIS 文件 (DEM, Lat, Lon)：从磁盘加载。
        """
        full_mask = np.full(self.default_shape, True, dtype=bool)
        
        # 1. 动态时间编码处理
        if var in [MetGis.MONTH_SIN, MetGis.MONTH_COS, MetGis.HOUR_SIN, MetGis.HOUR_COS]:
            try:
                obs_time = self.str_to_datetime(timestamp)
                raw_val = 0.0
                if var in [MetGis.MONTH_SIN, MetGis.MONTH_COS]:
                    # 月份周期性编码
                    angle = 2 * math.pi * float(obs_time.month) / 12.0
                    raw_val = math.sin(angle) if var == MetGis.MONTH_SIN else math.cos(angle)
                elif var in [MetGis.HOUR_SIN, MetGis.HOUR_COS]:
                    # 小时周期性编码
                    angle = 2 * math.pi * float(obs_time.hour) / 24.0
                    raw_val = math.sin(angle) if var == MetGis.HOUR_SIN else math.cos(angle)
                
                min_val, max_val = self._get_channel_limits(var)
                if max_val - min_val > 0:
                    normalized_value = (raw_val - min_val) / (max_val - min_val)
                else:
                    normalized_value = 0.5 
                
                normalized_value = max(0.0, min(1.0, normalized_value))
                # 扩展为全图常数
                data = np.full(self.default_shape, normalized_value, dtype=np.float32)
                return data, full_mask
            except Exception:
                return np.zeros(self.default_shape, dtype=np.float32), np.zeros(self.default_shape, dtype=bool)

        # 2. 静态 GIS 文件加载
        try:
            file_name = f"{var.value}.npy"
            file_path = os.path.join(self.gis_data_path, self.station_id, file_name)
            min_val, max_val = self._get_channel_limits(var)
            data, mask = self.normalize(file_path, min_val, max_val)
            
            if data is not None and mask is not None:
                data, mask = self._ensure_shape(data, mask)

            if data is None:
                return np.zeros(self.default_shape, dtype=np.float32), np.zeros(self.default_shape, dtype=bool)
            return data, mask

        except Exception:
            return np.zeros(self.default_shape, dtype=np.float32), np.zeros(self.default_shape, dtype=bool)

    def _preload_gis_data(self):
        """缓存静态 GIS 数据，避免在循环中重复执行 IO 读取。"""
        self._gis_cache.clear()
        default_data = np.zeros(self.default_shape, dtype=np.float32)
        default_mask = np.zeros(self.default_shape, dtype=bool)
        
        for channel in self.channels:
            if isinstance(channel, MetGis):
                cache_key = f"gis_{channel.value}"
                if cache_key not in self._gis_cache:
                    # 使用序列第一帧的时间戳即可（GIS通常是静态或仅依赖时间）
                    data, mask = self._load_gis_frame(channel, self.timestamps[0])
                    self._gis_cache[cache_key] = (data if data is not None else default_data, 
                                                 mask if mask is not None else default_mask)

    def _load_channel_frame(self, channel: Union[MetLabel, MetRadar, MetNwp, MetGis], timestamp: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """分发函数：根据通道类型枚举类调用对应的加载方法。"""
        loader_map = {
            MetLabel: self._load_label_frame,
            MetRadar: self._load_radar_frame,
            MetGis: self._load_gis_frame,
            MetNwp: self._load_nwp_frame,
        }

        loader = None
        for enum_cls, handler in loader_map.items():
            if isinstance(channel, enum_cls):
                loader = handler
                break

        if loader is None:
            return None, None

        data, mask = loader(channel, timestamp)

        if data is None:
            return None, None

        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask, dtype=bool)

        return data.astype(np.float32, copy=False), mask.astype(bool, copy=False)

    def _load_temporal_data_with_fallback(self, channel: Union[MetLabel, MetRadar, MetNwp, MetGis], timestamp: str, 
                                          timestep_idx: int, all_timestamps: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        时序数据加载的回退策略 (Fallback Strategy)。
        
        用于处理偶发的文件缺失情况。逻辑如下：
        1. 尝试加载当前时刻 (t)。
        2. 如果失败，尝试加载上一时刻 (t-1)。
        3. 如果失败，尝试加载下一时刻 (t+1)。
        4. 均失败则返回全零数据。
        
        注意：使用回退数据时，虽然填充了数值，但返回的掩码 (Mask) 仍为全 False，
        表示该时刻数据是“伪造”的，模型应当知晓其不可靠性。

        Args:
            timestep_idx (int): 当前时间步在 all_timestamps 中的索引。
            all_timestamps (List[str]): 完整的样本时间戳列表。

        Returns:
            Tuple[np.ndarray, np.ndarray]: (data, mask)。
        """
        default_data = np.zeros(self.default_shape, dtype=np.float32)
        default_mask = np.zeros(self.default_shape, dtype=bool)

        # 1. 尝试加载当前时刻
        data, mask = self._load_channel_frame(channel, timestamp)
        if data is not None and mask is not None:
            return data, mask
        
        # 2. 回退：前一时刻
        if timestep_idx > 0:
            prev_timestamp = all_timestamps[timestep_idx - 1]
            data, mask = self._load_channel_frame(channel, prev_timestamp)
            if data is not None:
                return data, default_mask
        
        # 3. 回退：后一时刻
        if timestep_idx < len(all_timestamps) - 1:
            next_timestamp = all_timestamps[timestep_idx + 1]
            data, mask = self._load_channel_frame(channel, next_timestamp)
            if data is not None:
                return data, default_mask
        
        return default_data, default_mask

    def _load_channel_frame_with_fallback(self, channel: Union[MetLabel, MetRadar, MetNwp, MetGis], timestamp: str, 
                                          timestep_idx: int, all_timestamps: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        统一的数据加载入口。
        
        集成了 GIS 缓存机制和动态时序数据的回退 (Fallback) 机制。
        
        Args:
            channel: 通道枚举。
            timestamp: 时间字符串。
            timestep_idx: 时间索引。
            all_timestamps: 时间列表。

        Returns:
            Tuple[np.ndarray, np.ndarray]: 加载并处理后的 (data, mask)。
        """
        default_data = np.zeros(self.default_shape, dtype=np.float32)
        default_mask = np.zeros(self.default_shape, dtype=bool)

        # 策略 A: GIS 数据优先查缓存
        if isinstance(channel, MetGis):
            cache_key = f"gis_{channel.value}"
            if cache_key in self._gis_cache:
                return self._gis_cache[cache_key]
            data, mask = self._load_gis_frame(channel, timestamp)
            if data is not None and mask is not None:
                self._gis_cache[cache_key] = (data, mask)
                return data, mask
            return default_data, default_mask
        
        # 策略 B: 动态时序数据 (Radar, NWP) 使用回退机制
        if isinstance(channel, (MetRadar, MetNwp)):
            return self._load_temporal_data_with_fallback(channel, timestamp, timestep_idx, all_timestamps)
        
        # 策略 C: 其他数据 (Label等) 直接加载，不回退
        data, mask = self._load_channel_frame(channel, timestamp)
        if data is not None and mask is not None:
            return data, mask
        
        return default_data, default_mask