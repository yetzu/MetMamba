# metai/utils/met_var.py

from enum import Enum
from typing import Any, Optional, Dict


class MetBaseEnum(str, Enum):
    """
    通用气象变量枚举基类。

    继承自 `str` 和 `Enum`，使其成员既可以作为字符串使用，又具备枚举的特性。
    同时扩展了元数据存储能力，为每个枚举成员绑定物理取值范围（最小值、最大值）和缺测值标记。
    这在数据预处理（如归一化）和异常值过滤中非常有用。

    Attributes:
        _min (Optional[float]): 变量的物理最小值。
        _max (Optional[float]): 变量的物理最大值。
        _missing_value (Optional[float]): 变量的缺测值标记（Missing Value）。
    """

    # 为类型检查器声明成员属性，使用 __slots__ 优化内存占用
    __slots__ = ("_min", "_max", "_missing_value")
    _min: Optional[float]
    _max: Optional[float]
    _missing_value: Optional[float]

    def __new__(cls, value: str, min_value: Optional[float] = None, max_value: Optional[float] = None, missing_value: Optional[float] = None):
        """
        初始化枚举成员。

        Args:
            value (str): 枚举成员的字符串标识（通常是变量名）。
            min_value (Optional[float], optional): 变量的物理最小值。默认为 None。
            max_value (Optional[float], optional): 变量的物理最大值。默认为 None。
            missing_value (Optional[float], optional): 原始数据中表示“缺失”的数值标记（如 -9999, -32768 等）。默认为 None。

        Returns:
            MetBaseEnum: 初始化后的枚举成员实例。
        """
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._min = min_value
        obj._max = max_value
        obj._missing_value = missing_value
        return obj

    @property
    def min(self) -> Optional[float]:
        """
        获取变量的物理最小值。

        Returns:
            Optional[float]: 最小值，若未定义则返回 None。
        """
        return self._min

    @property
    def max(self) -> Optional[float]:
        """
        获取变量的物理最大值。

        Returns:
            Optional[float]: 最大值，若未定义则返回 None。
        """
        return self._max

    @property
    def missing_value(self) -> Optional[float]:
        """
        获取变量的缺测值标记。

        Returns:
            Optional[float]: 缺测值，若未定义则返回 None。
        """
        return self._missing_value

    @property
    def limits(self) -> Optional[Dict[str, Optional[float]]]:
        """
        获取变量的数值范围和缺测配置字典。

        通常用于数据归一化（Min-Max Normalization）逻辑。

        Returns:
            Optional[Dict[str, Optional[float]]]: 包含 'min', 'max', 'missing_value' 的字典。
            如果 min 或 max 未定义，则返回 None。
        """
        if self._min is None or self._max is None:
            return None
        return {"min": self._min, "max": self._max, "missing_value": self._missing_value}

    @property
    def parent(self) -> str:
        """
        获取枚举所属的父类别名称（大类）。

        例如 MetRadar -> RADAR, MetNwp -> NWP。用于自动生成文件路径或日志标签。

        Returns:
            str: 去除 'Met' 前缀并大写的类名。
        """
        return self.__class__.__name__.replace("Met", "").upper()


class MetLabel(MetBaseEnum):
    """
    标签数据类型枚举 (Ground Truth)。

    用于定义气象预测任务中的目标变量。
    
    Enum Members:
        RA (tuple): 降水量 (Rainfall)，单位 mm。范围 [0, 300]，缺测值 -9。
    """
    RA = ("RA", 0, 300, -9)
    # RB = "RB"


class MetRadar(MetBaseEnum):
    """
    雷达数据类型枚举 (Radar Products)。

    包含各种双偏振或多普勒雷达观测产品。
    
    Enum Members:
        CR (tuple): 组合反射率 (Composite Reflectivity)，单位 dBZ。
        CAPxx (tuple): 特定高度层的等高平面位置显示反射率 (CAPPI)，例如 CAP20 表示 2km 高度。
        ET (tuple): 回波顶高 (Echo Top)，单位 km (或相关量纲)。
        HBR (tuple): 混合波束反射率 (Hybrid Scan Reflectivity)。
        VIL (tuple): 垂直液态水含量 (Vertically Integrated Liquid)，单位 kg/m²。
    """
    CR = ("CR", 0, 800, -32768)
    CAP20 = ("CAP20", 0, 800, -32768)
    CAP30 = ("CAP30", 0, 800, -32768)
    CAP40 = ("CAP40", 0, 800, -32768)
    CAP50 = ("CAP50", 0, 800, -32768)
    CAP60 = ("CAP60", 0, 800, -32768)
    CAP70 = ("CAP70", 0, 800, -32768)
    ET = ("ET", 0, 150, -1280)
    HBR = ("HBR", 0, 800, -32768)
    VIL = ("VIL", 0, 8000, -1280)


class MetNwp(MetBaseEnum):
    """
    数值天气预报数据类型枚举 (NWP Variables)。

    包含来自数值模式（如 GFS, ECMWF, GRAPES）的各种气象要素场。
    
    Enum Members:
        DVGxxx: 散度 (Divergence) @ xxx hPa。
        WSxxx: 风速 (Wind Speed) @ xxx hPa。
        Qxxx: 比湿 (Specific Humidity) @ xxx hPa。
        RHxxx: 相对湿度 (Relative Humidity) @ xxx hPa。
        PWAT: 大气可降水量 (Precipitable Water)。
        CAPE: 对流有效位能 (Convective Available Potential Energy)。
        ... 其他热力学与动力学参数。
    """
    DVG925 = ("DVG925", -1, 1, -9999)
    DVG850 = ("DVG850", -1, 1, -9999)
    DVG200 = ("DVG200", -1, 1, -9999)
    WS925 = ("WS925", 0, 30, -9999)
    WS700 = ("WS700", 0, 40, -9999)
    WS500 = ("WS500", 0, 50, -9999)
    Q1000 = ("Q1000", 0, 30, -9999)
    Q850 = ("Q850", 0, 20, -9999)
    Q700 = ("Q700", 0, 20, -9999)
    RH1000 = ("RH1000", 0, 100, -9999)
    RH700 = ("RH700", 0, 100, -9999)
    RH500 = ("RH500", 0, 100, -9999)
    PWAT = ("PWAT",0, 80, -9999)
    PE = ("PE",0, 100, -9999)
    TdSfc850 = ("TdSfc850", 0, 30, -9999)
    TTdMean74 = ("TTdMean74", 0, 30, -9999)
    TTdMax74 = ("TTdMax74", 0, 30, -9999)
    HTw0 = ("HTw0", 0, 3000, -9999)
    LCL = ("LCL", 0, 3000, -9999)
    muLCL = ("muLCL",0, 3000, -9999)
    KI = ("KI", 0, 40, -9999)
    LI500 = ("LI500", 0, 10000, -9999)
    LI300 = ("LI300", 0, 10000, -9999)
    HT0 = ("HT0", 0, 10000, -9999)
    HT10 = ("HT10", 0, 10000, -9999)
    LFC = ("LFC", 0, 10000, -9999)
    CAPE = ("CAPE", 0, 3000, -9999)


class MetGis(MetBaseEnum):
    """
    地理信息系统数据类型枚举 (GIS/Static Features)。

    包含地形数据和时间编码特征。
    
    Enum Members:
        LAT/LON: 经纬度网格。
        DEM: 数字高程模型 (Digital Elevation Model)。
        MONTH/HOUR: 原始月份和小时值。
        *_SIN/*_COS: 时间特征的周期性正弦/余弦编码 (Cyclic Encoding)，用于保持时间的连续性。
    """
    LAT = ("lat", 20, 50)
    LON = ("lon", 90, 130)
    DEM = ("dem", 0, 3000)
    MONTH = ("month", 0, 12)
    HOUR = ("hour", 0, 24)

    # 值域设为 -1 到 1，归一化时会映射到 [0, 1]
    MONTH_SIN = ("month_sin", -1, 1)
    MONTH_COS = ("month_cos", -1, 1)
    HOUR_SIN = ("hour_sin", -1, 1)
    HOUR_COS = ("hour_cos", -1, 1)


class MetVar:
    """
    气象变量注册表 (Variable Registry)。

    提供统一的访问接口来获取各种气象数据类型的枚举类。
    使用代理模式 (Proxy Pattern) 来组织不同类别的变量。

    Attributes:
        LABEL: 标签变量枚举代理 (指向 MetLabel)。
        RADAR: 雷达变量枚举代理 (指向 MetRadar)。
        NWP: 数值预报变量枚举代理 (指向 MetNwp)。
        GIS: 地理信息变量枚举代理 (指向 MetGis)。
    """

    class _MetVarAttr:
        """
        枚举属性代理类。
        
        用于拦截属性访问并转发给实际的 Enum 类，同时提供友好的字符串表示和哈希支持。
        """

        def __init__(self, enum_class: type, attr_name: str) -> None:
            """
            初始化代理对象。

            Args:
                enum_class (type): 被代理的枚举类 (如 MetRadar)。
                attr_name (str): 代理名称 (如 'RADAR')。
            """
            self.enum_class = enum_class
            self.name = attr_name

        def __getattr__(self, name: str) -> Any:
            """
            将属性访问转发至枚举类。

            Args:
                name (str): 枚举成员名称。

            Returns:
                Any: 对应的枚举成员。
            """
            return getattr(self.enum_class, name)

        def __repr__(self) -> str:
            """返回调试友好的字符串表示。"""
            return f"<{self.name} enum proxy: {self.enum_class.__name__}>"

        def __hash__(self) -> int:
            """
            计算哈希值。
            
            基于枚举类名和代理名称计算，确保代理对象在字典或集合中可作为键使用。
            """
            return hash((self.enum_class.__name__, self.name))

        def __eq__(self, other: Any) -> bool:
            """
            判断两个代理对象是否相等。

            Args:
                other (Any): 另一个对象。

            Returns:
                bool: 如果枚举类和名称均相同则返回 True。
            """
            if not isinstance(other, MetVar._MetVarAttr):
                return False
            return (self.enum_class is other.enum_class) and (self.name == other.name)

    # 各种气象数据类型的枚举代理
    LABEL = _MetVarAttr(MetLabel, "LABEL")
    RADAR = _MetVarAttr(MetRadar, "RADAR")
    NWP = _MetVarAttr(MetNwp, "NWP")
    GIS = _MetVarAttr(MetGis, "GIS")