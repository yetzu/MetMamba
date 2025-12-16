# run/infer_scwds_mamba.py
import sys
import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime, timedelta

# 设置 matplotlib 后端为 Agg (非交互模式)，适用于服务器环境
matplotlib.use('Agg')

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.utils import MetLabel, MLOGI, MLOGE
from metai.dataset.met_dataloader_scwds import ScwdsDataModule
from metai.model.mamba.trainer import MetMambaTrainer
from metai.model.mamba.config import ModelConfig
from metai.utils.met_config import get_config

# ==========================================
# 常量定义
# ==========================================
TRACK_ID = "track"       # 赛道ID
TIME_STEP_MINUTES = 6    # 预测时间步长 (分钟)
PHYSICAL_MAX_MM = 30.0   # 物理量最大值 (用于反归一化)

# ==========================================
# 辅助函数
# ==========================================

def find_latest_ckpt(save_dir: str) -> str:
    """在指定目录下查找最新的检查点文件 (.ckpt)。

    查找策略：
    1. 优先查找 'best.ckpt'。
    2. 其次查找 'last.ckpt'。
    3. 如果都不存在，递归搜索所有 .ckpt 文件并返回字典序最大的一个。

    Args:
        save_dir (str): 检查点保存的根目录。

    Returns:
        str: 找到的检查点文件绝对路径。

    Raises:
        FileNotFoundError: 如果未找到任何 .ckpt 文件。
    """
    best = os.path.join(save_dir, 'best.ckpt')
    if os.path.exists(best): return best
    
    last = os.path.join(save_dir, 'last.ckpt')
    if os.path.exists(last): return last
    
    # 递归搜索子目录
    search_pattern = os.path.join(save_dir, '**', '*.ckpt')
    cpts = sorted(glob.glob(search_pattern, recursive=True))
    
    if len(cpts) == 0:
        raise FileNotFoundError(f'No checkpoint found in {save_dir}')
    return cpts[-1]

def create_precipitation_cmap() -> tuple:
    """创建自定义降水色标 (Colormap)。

    定义符合气象业务习惯的降水色阶，用于可视化。

    Returns:
        tuple: (cmap, norm)
            - cmap (matplotlib.colors.ListedColormap): 颜色映射对象。
            - norm (matplotlib.colors.BoundaryNorm): 边界归一化对象。
    """
    hex_colors = [
        '#9CF48D', # 0.01 - 0.1 (微量/小雨)
        '#3CB73A', # 0.1 - 1.0
        '#63B7FF', # 1.0 - 2.0
        '#0200F9', # 2.0 - 5.0
        '#EE00F0', # 5.0 - 8.0
        '#9F0000'  # > 8.0 (暴雨)
    ]
    cmap = mcolors.ListedColormap(hex_colors)
    cmap.set_bad('white')   # 无效值颜色
    cmap.set_under('white') # 低于阈值颜色
    
    # 定义色标边界
    bounds = [0.01, 0.1, 1, 2, 5, 8, 100]
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=len(hex_colors))
    return cmap, norm

def plot_inference(obs_seq: np.ndarray, pred_seq: np.ndarray, save_path: str):
    """绘制推理结果对比图并保存。

    展示输入的历史观测序列和模型预测的未来序列。

    Args:
        obs_seq (np.ndarray): 历史观测序列 (物理量 mm)。
            Shape: [T_in, H, W]。
        pred_seq (np.ndarray): 预测序列 (物理量 mm)。
            Shape: [T_out, H, W]。
        save_path (str): 图片保存路径。
    """
    T_in = obs_seq.shape[0]
    T_out = pred_seq.shape[0]
    cols = max(T_in, T_out)
    
    # 获取自定义色标
    cmap, norm = create_precipitation_cmap()
    
    # 确保数据为物理量 (mm)
    obs_mm = obs_seq * PHYSICAL_MAX_MM
    pred_mm = pred_seq * PHYSICAL_MAX_MM
    
    fig, axes = plt.subplots(2, cols, figsize=(cols * 1.5, 3.5), constrained_layout=True)
    
    for t in range(cols):
        # 1. Input (Obs) Row
        ax = axes[0, t]
        if t < T_in:
            ax.imshow(obs_mm[t], cmap=cmap, norm=norm)
            if t == 0: ax.set_title('Obs (Past)', fontsize=10)
        else:
            ax.axis('off') # 补白
        ax.set_xticks([]); ax.set_yticks([])

        # 2. Output (Pred) Row
        ax = axes[1, t]
        if t < T_out:
            ax.imshow(pred_mm[t], cmap=cmap, norm=norm)
            if t == 0: ax.set_title('Pred (Future)', fontsize=10)
        else:
            ax.axis('off') # 补白
        ax.set_xticks([]); ax.set_yticks([])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(description='Infer SCWDS Mamba Model')
    parser.add_argument('--data_path', type=str, default='data/samples.testset.jsonl',
                        help='推理数据集路径 (.jsonl)')
    parser.add_argument('--in_shape', type=int, nargs=4, default=[10, 54, 256, 256],
                        help='输入张量形状 [T, C, H, W]')
    parser.add_argument('--save_dir', type=str, default='./output/mamba',
                        help='模型检查点所在目录')
    parser.add_argument('--accelerator', type=str, default='cuda',
                        help='推理设备 (cuda/cpu)')
    parser.add_argument('--vis', action='store_true', 
                        help='是否开启可视化输出')
    parser.add_argument('--vis_output', type=str, default='./output/mamba/vis_infer',
                        help='可视化结果保存目录')
    parser.add_argument('--ckpt_path', type=str, default=None, 
                        help='指定 checkpoint 路径。如果为 None，则在 save_dir 中自动搜索。')
    return parser.parse_args()

# ==========================================
# 主流程
# ==========================================

def main():
    """推理主程序。
    
    流程：
    1. 加载配置和模型检查点。
    2. 初始化推理数据加载器。
    3. 执行模型推理。
    4. 对结果进行后处理（插值、反归一化）。
    5. 保存符合竞赛格式的 .npy 文件。
    6. (可选) 生成可视化预览图。
    """
    args = parse_args()
    met_config = get_config() 
    
    # 1. Config 初始化
    config = ModelConfig(
        data_path=args.data_path,
        in_shape=tuple(args.in_shape),
        save_dir=args.save_dir,
    )
    device = torch.device(args.accelerator if torch.cuda.is_available() else 'cpu')
    FMT = met_config.file_date_format
    
    print(f"[INFO] Start Inference. Device: {device}, SaveDir: {args.save_dir}")

    # 2. Data 模块准备
    # 注意：ScwdsDataModule 会根据 stage='infer' 准备无标签数据
    data_module = ScwdsDataModule(
        data_path=config.data_path,
        batch_size=1,
        num_workers=1
    )
    data_module.setup('infer')
    infer_loader = data_module.infer_dataloader()
    
    # 3. Model 加载
    try:
        if args.ckpt_path is not None:
            # 显式指定
            if not os.path.exists(args.ckpt_path):
                raise FileNotFoundError(f"指定的 Checkpoint 文件不存在: {args.ckpt_path}")
            ckpt_path = args.ckpt_path
            MLOGI(f"加载指定检查点: {ckpt_path}")
        else:
            # 自动搜索
            ckpt_path = find_latest_ckpt(config.save_dir)
            MLOGI(f"加载自动搜索的检查点: {ckpt_path}")
            
        # 从 Checkpoint 恢复 MetMambaTrainer
        model = MetMambaTrainer.load_from_checkpoint(ckpt_path, map_location=device)
        model.eval().to(device)
    except Exception as e:
        MLOGE(f"模型加载失败: {e}")
        return

    # 4. Inference Loop (推理循环)
    with torch.no_grad():
        for bidx, (metadata_list, batch_x, input_mask) in enumerate(infer_loader):
            try:
                # -----------------------------------------------------------
                # 数据上载 GPU
                # batch_x shape: [B, T_in, C, H, W] (例如 [1, 10, 54, 256, 256])
                # -----------------------------------------------------------
                batch_x = batch_x.to(device)
                input_mask = input_mask.to(device) 
                
                # -----------------------------------------------------------
                # 模型推理
                # 调用 infer_step，输入为 tuple (metadata, x, mask)
                # Output batch_y shape: [B, T_out, C_out, H, W] (例如 [1, 20, 1, 256, 256])
                # -----------------------------------------------------------
                batch_y = model.infer_step((metadata_list, batch_x, input_mask), batch_idx=bidx)
                
                # 提取 Batch 中第一个样本，并取第 0 通道 (降水通道)
                # Shape change: [1, 20, 1, 256, 256] -> [20, 256, 256]
                batch_y = batch_y[0, :, 0, :, :] 
                
                # 解析元数据
                metadata = metadata_list[0]
                sample_id = metadata['sample_id']
                
                # 从 sample_id 或 metadata 解析字段用于文件名生成
                # ID 格式示例: {task_id}_{region_id}_{time_id}_{station_id}
                sample_id_parts = sample_id.split('_')
                task_id = metadata.get('task_id')
                region_id = metadata['region_id']
                station_id = metadata['station_id']
                time_id = sample_id_parts[2] if len(sample_id_parts) > 2 else "000000000000"
                case_id = metadata.get('case_id')
                timestamps = metadata['timestamps'] # 历史观测的时间戳列表

                if not timestamps: continue
                
                # 计算起始预测时间 (基于最后一个观测时间点)
                last_obs_idx = batch_x.shape[1] - 1
                if last_obs_idx >= len(timestamps):
                    last_obs_idx = -1
                
                try:
                    last_obs_dt = datetime.strptime(timestamps[last_obs_idx], FMT)
                except ValueError:
                    # 如果时间格式解析失败，回退到当前时间 (仅作为防御性编程)
                    last_obs_dt = datetime.now() 

                pred_frames_vis = []
                seq_max_val = 0.0
                seq_mean_val = 0.0
                seq_zero_count = 0
                seq_total_count = 0
                
                # -----------------------------------------------------------
                # 逐帧后处理与保存
                # -----------------------------------------------------------
                for idx, y in enumerate(batch_y):
                    # 1. 空间分辨率插值 (Upsample)
                    # 竞赛要求输出分辨率通常可能与模型输入不同 (如 301x301 vs 256x256)
                    # Input y: [256, 256] -> [1, 1, 256, 256] -> Interpolate -> [301, 301]
                    y_interp = F.interpolate(
                        y.unsqueeze(0).unsqueeze(0), 
                        size=(301, 301),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze() # -> [301, 301]
                    
                    y_np = y_interp.cpu().numpy()

                    # 2. 反归一化 (Normalized -> Physical mm)
                    # MetLabel.RA.max 通常为 100.0 或其他缩放因子，这里需根据训练配置确认
                    # 转换为 float32，保留精度
                    y_save = (y_np * MetLabel.RA.max).astype(np.float32)
                    
                    # 3. 阈值过滤 (去噪)
                    # 小于 1.0 (这里的单位可能是 0.1mm 或其他，视 MetLabel 定义) 的值置 0
                    y_save[y_save < 1.0] = 0.0
                    
                    # 4. 生成文件名
                    forecast_dt = last_obs_dt + timedelta(minutes=TIME_STEP_MINUTES * (idx + 1))
                    forecast_time_str = forecast_dt.strftime(FMT)
                    
                    npy_dir = os.path.join('submit', 'output', TRACK_ID, case_id)
                    os.makedirs(npy_dir, exist_ok=True)
                    
                    npy_path = os.path.join(
                        npy_dir,
                        f"{task_id}_{region_id}_{time_id}_{station_id}_Forcast_{forecast_time_str}.npy"
                    )
                    
                    # 5. 保存 .npy
                    np.save(npy_path, y_save)
                    
                    # 统计信息更新 (用于日志)
                    # 假设 y_save 的单位是 0.1mm，除以 10 还原为 mm
                    seq_max_val = max(seq_max_val, float(y_save.max()) / 10.0)
                    seq_mean_val += float(y_save.mean()) / 10.0
                    seq_zero_count += np.sum(y_save == 0)
                    seq_total_count += y_save.size
                    
                    # 收集可视化数据 (0-1 范围，除以 300 是假设最大值为 30mm * 10)
                    if args.vis:
                        pred_frames_vis.append(y_save / 300.0)
                
                # 打印单样本统计
                seq_mean_val /= len(batch_y)
                zero_ratio = seq_zero_count / seq_total_count if seq_total_count > 0 else 0.0
                MLOGI(f"No.{bidx} {sample_id} | Max: {seq_max_val:.2f}mm | Mean: {seq_mean_val:.4f}mm | Zero: {zero_ratio:.2%}")

                # -----------------------------------------------------------
                # 可视化 (Optional)
                # -----------------------------------------------------------
                if args.vis:
                    # batch_x 维度 [1, 10, 54, 256, 256]，取第 0 通道 (Radar/Precip)
                    obs_frames = batch_x[0, :, 0, :, :].cpu().numpy()
                    pred_frames = np.array(pred_frames_vis)
                    vis_path = os.path.join(args.vis_output, f"{sample_id}.png")
                    plot_inference(obs_frames, pred_frames, vis_path)

            except Exception as e:
                import traceback
                traceback.print_exc()
                MLOGE(f"样本 {bidx} 推理失败: {e}")
                continue
            
    MLOGI("推理全部完成！")

if __name__ == '__main__':
    main()