# metai/dataset/met_case.py

import os
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime, timedelta
from metai.utils import MetLabel, MetRadar, MetNwp
from metai.utils.met_config import get_config, MetConfig

@dataclass
class MetCase:
    """
    天气个例 (Case) 管理类。

    该类基于竞赛数据的目录结构设计，用于统一管理单个天气过程（Case）的元数据、
    文件路径以及数据完整性校验逻辑。支持从 Case ID 解析任务信息，并提供
    针对标签 (Label)、雷达 (Radar) 和数值预报 (NWP) 文件的加载与时序校验功能。

    Attributes:
        case_id (str): 个例唯一标识符 (例如: 'CP_00_08220804_00093')。
        task_id (str): 任务类别标识。'CP' (短时强降水), 'TSW' (雷暴大风), 'HA' (冰雹)。
        region_id (str): 区域编码 (例如: 'AH' 安徽, 'CP' 综合)。
        station_id (str): 观测站点代码 (例如: 'Z9796', '00093')。
        base_path (str): 该个例数据集在文件系统中的根目录路径。
        radar_type (str): 该个例对应的雷达型号 ('SA', 'SB', 'SC')。默认为 'SA'，会在校验时自动更新。
        is_train (bool): 是否属于训练集。默认为 True。
        test_set (str): 测试集目录名称，仅在 is_train=False 时有效。默认为 "TestSetB"。
        label_files (List[str]): 初始化后自动加载的该个例下所有标签文件名列表。
    """
    case_id: str # 个例唯一标识符(如: CP_00_08220804_00093)
    
    # 业务属性
    task_id: str        # 任务类别: CP(短时强降水), TSW(雷暴大风), HA(冰雹)
    region_id: str      # 区域编码 (如: AH, CP)
    station_id: str     # 雷达站点代码 (如: Z9796, 00093)
    base_path: str      # 天气个例数据集根目录路径
    radar_type: str = 'SA' # 雷达类型: SA, SB, SC

    # 数据集类型配置
    is_train: bool = True        # 是否为训练集
    test_set: str = "TestSetB"      # 测试集类型: "TestSet" 或 "TestSetB"
    
    def __post_init__(self):
        """初始化后自动加载标签文件列表。"""
        self.label_files = self._load_label_files()
    
    @classmethod
    def create(cls, case_id: str, config: Optional['MetConfig'] = None, **kwargs) -> 'MetCase':
        """
        工厂方法：根据 Case ID 创建 MetCase 实例。

        解析 `case_id` 中的字段结构 (Task_Region_Time_Station)，结合全局配置
        自动构建个例的完整文件路径。

        Args:
            case_id (str): 个例唯一标识符 (格式: 任务_区域_时间_站点)。
            config (Optional[MetConfig]): 配置对象。若为 None，则自动获取全局默认配置。
            **kwargs: 传递给构造函数的其他参数 (如 `is_train`, `test_set`)。

        Returns:
            MetCase: 初始化完成的个例对象。

        Raises:
            ValueError: 当 `case_id` 格式不符合预期 (非4段式) 时抛出。
        """
        parts = case_id.split('_')
        if len(parts) != 4:
            raise ValueError(f"[ERROR] Invalid case_id format: {case_id}")
        
        task_id, region_id, _, station_id = parts

        # 获取配置，如果未提供则自动获取
        if config is None:
            config = get_config()
        
        # 构建个例根路径：Root/Task/DatasetType/Region/CaseID
        base_path = os.path.join(
            config.root_path,
            task_id,
            "TrainSet" if kwargs.get('is_train', True) else kwargs.get('test_set', "TestSetB"),
            region_id,
            case_id,
        )
        
        return cls(
            case_id=case_id,
            task_id=task_id,
            region_id=region_id,
            station_id=station_id,
            base_path=base_path,
            **kwargs
        )
    
    def _load_label_files(self, label_var: MetLabel = MetLabel.RA, return_full_path: bool = False) -> List[str]:
        """
        加载当前个例指定标签变量的所有文件列表。

        Args:
            label_var (MetLabel): 标签变量类型 (例如 MetLabel.RA)。默认为 RA (降水)。
            return_full_path (bool): 是否返回绝对路径。True 返回完整路径，False 仅返回文件名。

        Returns:
            List[str]: 按文件名排序后的标签文件路径列表。若目录不存在或加载失败返回空列表。
        """
        
        data_dir = os.path.join(
            self.base_path,
            "LABEL",
            label_var.name
        )
        try:
            if return_full_path:
                return sorted([os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.npy')])
            else:
                return sorted([file for file in os.listdir(data_dir) if file.endswith('.npy')])  
        except Exception as e:
            print(f"[ERROR] 加载标签文件失败: {e}")
            return []
    
    def _load_radar_files(self, radar_var: MetRadar = MetRadar.CR, return_full_path: bool = False) -> List[str]:
        """
        加载当前个例指定雷达产品的所有文件列表。

        Args:
            radar_var (MetRadar): 雷达产品类型 (例如 MetRadar.CR, MetRadar.CAP20)。
            return_full_path (bool): 是否返回绝对路径。

        Returns:
            List[str]: 按文件名排序后的雷达文件路径列表。
        """
        
        data_dir = os.path.join(
            self.base_path,
            "RADAR",
            radar_var.value
        )
        
        try:
            if return_full_path:
                return sorted([os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.npy')])
            else:
                return sorted([file for file in os.listdir(data_dir) if file.endswith('.npy')])
        except Exception as e:
            print(f"[ERROR] 加载雷达文件失败: {e}")
            return []

    def _load_nwp_files(self, nwp_type: MetNwp = MetNwp.CAPE, return_full_path: bool = False) -> List[str]:
        """
        加载当前个例指定数值预报 (NWP) 变量的所有文件列表。

        Args:
            nwp_type (MetNwp): NWP 变量类型 (例如 MetNwp.CAPE, MetNwp.WS850)。
            return_full_path (bool): 是否返回绝对路径。

        Returns:
            List[str]: 按文件名排序后的 NWP 文件路径列表。
        """
        
        data_dir = os.path.join(
            self.base_path,
            "NWP",
            nwp_type.name
        )

        try:
            if return_full_path:
                return sorted([os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.npy')])
            else:
                return sorted([file for file in os.listdir(data_dir) if file.endswith('.npy')])
        except Exception as e:
            print(f"[ERROR] 加载数值预报文件失败: {e}")
            return []

    def _extract_timestamp_from_label_file(self, filename: str) -> Optional[datetime]:
        """
        从标签文件名中提取时间戳。

        文件名格式通常为: `Task_Label_Var_Station_Time.npy`
        例如: `CP_Label_RA_00093_0511-1712.npy`

        Args:
            filename (str): 标签文件名。

        Returns:
            Optional[datetime]: 解析成功返回 datetime 对象，失败返回 None。
        """
        name_without_ext = filename.replace('.npy', '')
        parts = name_without_ext.split('_')
        if len(parts) >= 4:
            date_time = parts[-1]
            config = get_config()
            
            # 使用配置的日期格式 (例如 "%m%d-%H%M") 进行解析
            date_format = config.get_date_format()
            try:
                return datetime.strptime(date_time, date_format)
            except ValueError:
                return None
        
        return None

    def _validate_label_time_consistency(self, max_interval_minutes: int = 10) -> List[List[str]]:
        """
        全量标签文件的时间连续性校验。

        检查当前个例下所有标签文件，若相邻文件时间间隔超过 `max_interval_minutes`，
        则认为序列中断。

        Args:
            max_interval_minutes (int): 允许的最大时间间隔（分钟）。默认 10 分钟。

        Returns:
            List[List[str]]: 返回切分后的连续文件序列列表。
                例如: [[file1, file2], [file4, file5, file6]] (中间断开了 file3)。
        """
        label_files = self._load_label_files(return_full_path=True)
        
        if len(label_files) < 2:
            return [label_files] if label_files else [[]]
        
        # 扫描时间断点
        split_indices = []
        for i in range(len(label_files) - 1):
            timestamp1 = self._extract_timestamp_from_label_file(os.path.basename(label_files[i]))
            timestamp2 = self._extract_timestamp_from_label_file(os.path.basename(label_files[i + 1]))
            
            if timestamp1 and timestamp2:
                interval_minutes = (timestamp2 - timestamp1).total_seconds() / 60
                if interval_minutes > max_interval_minutes:
                    split_indices.append(i + 1)
        
        # 根据断点切分序列
        if not split_indices:
            return [label_files]
        
        split_arrays = []
        start_idx = 0
        
        for split_idx in split_indices:
            split_arrays.append(label_files[start_idx:split_idx])
            start_idx = split_idx
        
        # 添加最后一段
        if start_idx < len(label_files):
            split_arrays.append(label_files[start_idx:])
        
        return split_arrays
    
    def _validate_label_time_consistency_for_files(self, label_files: List[str], max_interval_minutes: int = 10) -> List[List[str]]:
        """
        指定标签文件列表的时间连续性校验与切分。

        Args:
            label_files (List[str]): 待校验的标签文件路径列表。
            max_interval_minutes (int): 允许的最大时间间隔（分钟）。

        Returns:
            List[List[str]]: 切分后的连续文件子序列列表。
        """
        if len(label_files) < 2:
            return [label_files] if label_files else [[]]
        
        # 解析时间戳并排序，确保输入顺序混乱时不影响逻辑
        files_with_timestamps = []
        for file_path in label_files:
            timestamp = self._extract_timestamp_from_label_file(os.path.basename(file_path))
            if timestamp:
                files_with_timestamps.append((file_path, timestamp))
        
        # 按时间升序排列
        files_with_timestamps.sort(key=lambda x: x[1])
        sorted_files = [file_path for file_path, _ in files_with_timestamps]
        
        # 扫描断点
        split_indices = []
        for i in range(len(sorted_files) - 1):
            timestamp1 = self._extract_timestamp_from_label_file(os.path.basename(sorted_files[i]))
            timestamp2 = self._extract_timestamp_from_label_file(os.path.basename(sorted_files[i + 1]))
            
            if timestamp1 and timestamp2:
                interval_minutes = (timestamp2 - timestamp1).total_seconds() / 60
                if interval_minutes > max_interval_minutes:
                    split_indices.append(i + 1)
        
        # 切分
        if not split_indices:
            return [sorted_files]
        
        split_arrays = []
        start_idx = 0
        
        for split_idx in split_indices:
            split_arrays.append(sorted_files[start_idx:split_idx])
            start_idx = split_idx
        
        if start_idx < len(sorted_files):
            split_arrays.append(sorted_files[start_idx:])
        
        return split_arrays
    
    def _is_radar_file_valid(self, obsdate: datetime, radar_var: MetRadar) -> bool:
        """
        校验特定时间点和类型的雷达文件是否存在。

        同时会根据目录下第一个文件自动推断并更新 `self.radar_type`。

        Args:
            obsdate (datetime): 观测时间。
            radar_var (MetRadar): 雷达变量类型。

        Returns:
            bool: 文件存在返回 True，否则返回 False。
        """
        file_directory = os.path.join(
            self.base_path,
            "RADAR",
            radar_var.value
        )

        try:
            if not os.path.exists(file_directory):
                return False
                
            # 动态检测雷达型号 (SA/SB/SC)，仅在第一次调用时生效
            file_names = sorted([file for file in os.listdir(file_directory) if file.endswith('.npy')])

            if len(file_names):
                # 假设格式: Task_RADA_Station_Time_RadarType_Var.npy
                self.radar_type = file_names[0].split('_')[-2]
            else:
                return False
            
            # 构造预期文件名并检查是否存在
            config = get_config()
            date_format = config.get_date_format()
            filename = '_'.join([self.task_id, 'RADA', self.station_id, obsdate.strftime(date_format), self.radar_type, radar_var.value]) + ".npy"
            
            file_path = os.path.join(
                file_directory,
                filename
            )
            
            return os.path.exists(file_path)
        except Exception as e:
            print(f"[ERROR] 验证雷达文件失败: {e}")
            return False

    def _validate_radar_completeness(self, label_file: str) -> bool:
        """
        校验单个样本（Label）对应的全要素雷达数据完整性。

        要求该 Label 对应时刻的所有 `MetRadar` 枚举定义的雷达产品文件均存在。

        Args:
            label_file (str): 标签文件路径 (作为基准时间)。

        Returns:
            bool: 所有雷达文件均存在返回 True，否则返回 False。
        """
        obsdate = self._extract_timestamp_from_label_file(os.path.basename(label_file))
        if not obsdate:
            return False
        
        radar_vars = list(MetRadar)
        for radar_var in radar_vars:
            if not self._is_radar_file_valid(obsdate, radar_var):
                return False
        return True
    
    def _is_nwp_file_valid(self, obsdate: datetime, nwp_var: MetNwp) -> bool:
        """
        校验特定时间点和类型的 NWP 文件是否存在。

        包含时间对齐逻辑：NWP 数据通常为逐小时，而 Label 为逐6分钟。
        策略：分钟数 < 30 向下取整到整点，>= 30 向上取整到下一整点。

        Args:
            obsdate (datetime): 观测时间。
            nwp_var (MetNwp): NWP 变量类型。

        Returns:
            bool: 文件存在返回 True。
        """
        file_directory = os.path.join(
            self.base_path,
            "NWP",
            nwp_var.name,
        )

        # 时间对齐逻辑 (Nearest Neighbor Temporal Interpolation)
        if obsdate.minute < 30:
            obsdate = obsdate.replace(minute=0)
        else:
            obsdate = obsdate.replace(minute=0) + timedelta(hours=1)

        try:
            if not os.path.exists(file_directory):
                return False
            
            config = get_config()
            date_format = config.get_date_format()
            nwp_prefix = config.get_nwp_prefix()
            
            # 构造预期文件名: Task_Prefix_Station_Time_Var.npy
            filename = '_'.join([self.task_id, nwp_prefix, self.station_id, obsdate.strftime(date_format), nwp_var.value]) + ".npy"
            
            file_path = os.path.join(
                file_directory,
                filename
            )
            
            return os.path.exists(file_path)
        except Exception as e:
            print(f"[ERROR] 验证数值预报文件失败: {e}")
            return False
        
    def _validate_nwp_completeness(self, label_file: str) -> bool:
        """
        校验单个样本（Label）对应的全要素 NWP 数据完整性。

        Args:
            label_file (str): 标签文件路径。

        Returns:
            bool: 所有 NWP 文件均存在返回 True。
        """
        obsdate = self._extract_timestamp_from_label_file(os.path.basename(label_file))
        if not obsdate:
            return False
        
        nwp_vars = list(MetNwp)
        for nwp_var in nwp_vars:
            if not self._is_nwp_file_valid(obsdate, nwp_var):
                return False
        return True

    def get_valid_sequences(self, min_length: int = 40, max_interval_minutes: int = 10) -> List[List[str]]:
        """
        获取当前个例中所有符合条件的连续数据序列。

        处理流程：
        1. 完整性筛选: 剔除雷达或 NWP 数据缺失的 Label 时刻。
        2. 连续性校验: 对剩余有效 Label 按时间间隔进行切分，得到若干连续子序列。

        Args:
            min_length (int): 序列最小长度。小于此长度的序列将被丢弃。默认 40 (对应 4 小时数据)。
            max_interval_minutes (int): 判定序列中断的时间间隔阈值。默认 10 分钟。

        Returns:
            List[List[str]]: 有效序列列表，每个元素为一个由文件路径组成的列表。
        """
        # 第一步：数据完整性检验 (Completeness Check)
        valid_files = []
        for file in self.label_files:
            if self._validate_radar_completeness(file) and self._validate_nwp_completeness(file):
                valid_files.append(file)
                
        # 第二步：时间一致性检验与切分 (Time Consistency Check)
        valid_sequences = []
        if len(valid_files) >= min_length:
            split_arrays = self._validate_label_time_consistency_for_files(
                valid_files, max_interval_minutes=max_interval_minutes
            )
            
            # 过滤短序列
            for sequence in split_arrays:
                if len(sequence) >= min_length:
                    valid_sequences.append(sequence)
        
        return valid_sequences

    def to_samples(self, sample_length: int = 40, sample_interval: int = 10) -> List[List[str]]:
        """
        生成训练样本（滑动窗口）。

        将通过完整性与连续性校验的序列，按固定步长滑动切割成定长样本。

        Args:
            sample_length (int): 单个样本的时间序列长度。默认 40 帧。
            sample_interval (int): 滑动窗口的步长 (Stride)。默认 10 帧。

        Returns:
            List[List[str]]: 样本列表。每个样本包含 `sample_length` 个文件路径。
        """
        # 获取有效序列，放宽间隔以允许在序列内部进行采样，但序列生成逻辑本身通常使用严格间隔
        # 此处 max_interval_minutes=20 可能是为了容忍少量缺失帧，视具体业务逻辑而定
        valid_sequences =  self.get_valid_sequences(min_length=sample_length, max_interval_minutes=20)

        samples = []
        
        for sequence in valid_sequences:
            if len(sequence) < sample_length:
                continue
                
            # 滑动窗口采样
            for start_idx in range(0, len(sequence) - sample_length + 1, sample_interval):
                end_idx = start_idx + sample_length
                sample = sequence[start_idx:end_idx]
                samples.append(sample)
        
        return samples
    
    def to_infer_sample(self, sample_length: int = 20) -> List[List[str]]:
        """
        获取推理样本。

        通常提取个例中最新的 `sample_length` 帧数据用于预测未来。

        Args:
            sample_length (int): 需要的输入序列长度。默认 20 帧。

        Returns:
            List[List[str]]: 包含单个样本的列表，或者为空 (如果数据不足)。
        """
        lable_files = self._load_label_files(return_full_path=True)
        
        if len(lable_files) < sample_length:
            return []
        
        # 截取最后 N 帧
        seq = lable_files[-sample_length:]
        
        return [seq]