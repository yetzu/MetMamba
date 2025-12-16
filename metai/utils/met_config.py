# metai/utils/met_config.py

import yaml
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

"""
MetAI 配置管理模块。

提供统一的配置加载、保存与校验机制，支持：
- YAML 格式配置文件的读写操作
- 默认配置生成与全局单例模式 (Singleton Pattern) 管理
- 关键配置项的有效性校验（如路径存在性、日期格式合法性）
"""

@dataclass
class MetConfig:
    """
    MetAI 基础配置类 (Data Class)。

    用于管理项目运行所需的静态环境配置，包括数据集路径、文件命名规则及地理信息路径等。
    该类通常作为全局单例使用，确保系统各模块访问一致的配置上下文。

    Attributes:
        root_path (str): 数据集根目录的绝对路径。
        gis_data_path (str): GIS 地理信息数据（如 DEM 高程数据）的存储路径。
        file_date_format (str): 文件名中的日期时间格式字符串，需符合 `datetime.strftime` 规范。
        nwp_prefix (str): 数值天气预报 (NWP) 文件的命名前缀标识。
    """
    
    # 数据集根目录路径
    root_path: str = "/home/dataset-local/SevereWeather_AI_2025"
    
    # GIS / DEM 地理数据路径
    gis_data_path: str = "/home/dataset-local/SevereWeather_AI_2025/dem"

    # 文件日期格式（`datetime.strftime` 格式）
    file_date_format: str = "%m%d-%H%M"

    # NWP 文件前缀
    nwp_prefix: str = "RRA"
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'MetConfig':
        """
        从 YAML 配置文件实例化配置对象。

        Args:
            config_path (Union[str, Path]): YAML 配置文件的文件系统路径。

        Returns:
            MetConfig: 解析后的配置对象实例。若文件不存在或解析异常，则返回默认配置实例。
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            print(f"[ERROR] 配置文件不存在: {config_path}")
            return cls()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # 创建配置对象并动态更新属性
            config = cls()
            for key, value in data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    print(f"[ERROR] 配置文件包含未知配置项: {key}，已忽略。")
            
            return config
            
        except Exception as e:
            print(f"[ERROR] 读取配置文件失败: {e}")
            return cls()
    
    @classmethod
    def load(cls, config_path: Optional[Union[str, Path]] = None) -> 'MetConfig':
        """
        加载系统配置。

        该方法实现了配置文件的自动查找逻辑。如果未指定路径，则按照预定义的优先级顺序查找 `config.yaml`。

        Args:
            config_path (Optional[Union[str, Path]]): 指定的配置文件路径。
                若为 None，则按以下优先级查找：
                1. 当前工作目录 (`./config.yaml`)
                2. 当前目录下的 metai 子目录 (`./metai/config.yaml`)
                3. 代码库根目录 (基于文件位置推断)

        Returns:
            MetConfig: 加载完成的配置对象。
        """
        # 解析配置文件路径
        if config_path is None:
            config_file = "config.yaml"
            
            # 默认查找顺序（按优先级从高到低）
            default_paths = [
                Path.cwd() / config_file,
                Path.cwd() / "metai" / config_file,
                Path(__file__).parent.parent / config_file,
            ]
            
            for path in default_paths:
                if path.exists():
                    config_path = path
                    break
        
        if config_path:
            config = cls.from_file(config_path)
            print(f"[INFO] 已从以下路径加载配置: {config_path}")
        else:
            config = cls()
            print("[INFO] 未找到配置文件，使用默认配置初始化。")
        
        return config
    
    def save(self, config_path: Union[str, Path]) -> bool:
        """
        将当前配置状态持久化保存为 YAML 文件。

        Args:
            config_path (Union[str, Path]): 目标保存路径。如果父目录不存在，将自动创建。

        Returns:
            bool: 保存成功返回 True，失败返回 False。
        """
        config_path = Path(config_path)
        
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 序列化：导出非私有、非可调用属性
            config_data = {}
            for attr_name in dir(self):
                if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                    config_data[attr_name] = getattr(self, attr_name)
            
            # 写入 YAML 文件
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
            print(f"[INFO] 配置已成功保存至: {config_path}")
            return True
            
        except Exception as e:
            print(f"[ERROR] 保存配置文件失败: {e}")
            return False
    
    def get_root_path(self) -> str:
        """获取数据集根目录路径。"""
        return self.root_path
    
    def get_date_format(self) -> str:
        """获取文件日期格式字符串。"""
        return self.file_date_format
    
    def get_nwp_prefix(self) -> str:
        """获取 NWP 数据文件前缀。"""
        return self.nwp_prefix
    
    def validate(self) -> bool:
        """
        执行配置项的逻辑校验。

        检查关键配置（如路径非空、日期格式合法性）是否符合运行要求。

        Returns:
            bool: 校验通过返回 True；否则返回 False 并打印错误详情。
        """
        errors = []
        
        # 必填项校验
        if not self.root_path:
            errors.append("root_path (数据集根目录) 不能为空")
        
        # 日期格式校验
        try:
            from datetime import datetime
            test_date = datetime(2024, 1, 1, 12, 0)
            test_date.strftime(self.file_date_format)
        except ValueError as e:
            errors.append(f"file_date_format 日期格式无效: {e}")
        
        if errors:
            for error in errors:
                print(f"[ERROR] 配置校验失败: {error}")
            return False
        
        return True


# 全局配置实例（懒加载）
_config: Optional[MetConfig] = None


def get_config(config_path: Optional[Union[str, Path]] = None) -> MetConfig:
    """
    获取全局配置单例（懒加载模式）。

    如果全局实例尚未初始化，或者传入了新的 `config_path`，则会触发配置加载流程。
    
    Args:
        config_path (Optional[Union[str, Path]]): 配置文件路径。若为 None，则尝试自动查找或复用现有实例。
        
    Returns:
        MetConfig: 全局唯一的配置对象实例。
    """
    global _config
    
    # 如果显式指定了 config_path，或尚未加载，则重新加载配置
    if config_path is not None or _config is None:
        _config = MetConfig.load(config_path)
        # 加载后立即校验，若校验失败则回退至默认配置
        if not _config.validate():
            print("[ERROR] 配置校验未通过，回退至默认配置。")
            _config = MetConfig()
    
    return _config


def reload_config(config_path: Optional[Union[str, Path]] = None) -> MetConfig:
    """
    强制重新加载配置。

    无论全局实例是否存在，均强制重新读取配置文件并更新全局实例。
    
    Args:
        config_path (Optional[Union[str, Path]]): 配置文件路径。
        
    Returns:
        MetConfig: 重新加载后的新配置对象。
    """
    global _config
    _config = MetConfig.load(config_path)
    return _config


def create_default_config(config_path: Union[str, Path]) -> bool:
    """
    生成并保存默认配置文件模板。
    
    Args:
        config_path (Union[str, Path]): 保存路径。
        
    Returns:
        bool: 创建成功返回 True。
    """
    config = MetConfig()
    return config.save(config_path)