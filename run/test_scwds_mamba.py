# run/test_scwds_mamba.py
import sys
import os
import glob
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
matplotlib.use('Agg')

from metai.dataset.met_dataloader_scwds import ScwdsDataModule
from metai.model.mamba.trainer import MetMambaTrainer
from metai.model.mamba.metrices import MetScore

# ==========================================
# Part 0: 辅助工具函数
# ==========================================

class TeeLogger:
    """双向日志记录器，同时将输出重定向到控制台和文件。

    用于在长时间运行的任务中保留完整的终端输出记录。

    Attributes:
        log_file (file object): 日志文件句柄。
        console (file object): 标准输出句柄 (sys.stdout)。
    """

    def __init__(self, log_file_path: str):
        """初始化日志记录器。

        Args:
            log_file_path: 日志文件的保存路径。
        """
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.console = sys.stdout
        
    def write(self, message: str):
        """写入消息。

        Args:
            message: 需要写入的字符串内容。
        """
        self.console.write(message)
        self.log_file.write(message)
        self.log_file.flush()
        
    def flush(self):
        """刷新缓冲区，确保内容即时写入。"""
        self.console.flush()
        self.log_file.flush()
        
    def close(self):
        """关闭日志文件句柄。"""
        if self.log_file:
            self.log_file.close()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# 全局日志对象占位符
_logger = None

def set_logger(log_file_path: str):
    """设置全局日志重定向。

    Args:
        log_file_path: 日志文件路径。
    """
    global _logger
    _logger = TeeLogger(log_file_path)
    sys.stdout = _logger
    
def restore_stdout():
    """恢复标准输出（sys.stdout）到默认控制台。"""
    global _logger
    if _logger:
        sys.stdout = _logger.console
        _logger.close()
        _logger = None

def find_best_ckpt(save_dir: str) -> str:
    """在指定目录中递归查找最佳或最新的检查点文件 (.ckpt)。

    查找策略：
    1. 优先查找 explicitly named 'best.ckpt' 或 'last.ckpt'.
    2. 如果未找到，递归搜索所有 .ckpt 文件，并按字典序排序返回最后一个（通常对应最大的 epoch）。

    Args:
        save_dir: 检查点保存的根目录。

    Returns:
        str: 找到的检查点文件的绝对路径。

    Raises:
        FileNotFoundError: 如果在目录下未找到任何 .ckpt 文件。
    """
    # 0. 如果传入的是具体文件路径，直接返回
    if os.path.isfile(save_dir):
        return save_dir

    # 1. 优先查找根目录下的 best.ckpt 或 last.ckpt
    best = os.path.join(save_dir, 'best.ckpt')
    if os.path.exists(best): return best
    
    last = os.path.join(save_dir, 'last.ckpt')
    if os.path.exists(last): return last
    
    # 2. 递归查找所有 .ckpt 文件
    print(f"[INFO] Searching for checkpoints in {save_dir} recursively...")
    search_pattern = os.path.join(save_dir, '**', '*.ckpt')
    all_cpts = glob.glob(search_pattern, recursive=True)
    
    if not all_cpts:
        raise FileNotFoundError(f'No checkpoint found in {save_dir}')

    # 3. 排序并返回最后一个 (通常是 epoch 最大的)
    all_cpts = sorted(all_cpts)
    
    found_ckpt = all_cpts[-1]
    print(f"[INFO] Found checkpoint: {found_ckpt}")
    return found_ckpt

def get_checkpoint_info(ckpt_path: str) -> dict:
    """读取 Checkpoint 文件头信息，不加载模型权重。

    Args:
        ckpt_path: .ckpt 文件路径。

    Returns:
        dict: 包含 'epoch', 'global_step', 'hparams' 等元数据的字典。
              如果读取失败，返回包含 'error' 键的字典。
    """
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        epoch = ckpt.get('epoch', None)
        global_step = ckpt.get('global_step', None)
        hparams = ckpt.get('hyper_parameters', {})
        return {
            'epoch': epoch,
            'global_step': global_step,
            'hparams': hparams,
            'ckpt_name': os.path.basename(ckpt_path)
        }
    except Exception as e:
        return {'error': str(e)}

def print_checkpoint_info(ckpt_info: dict):
    """格式化打印 Checkpoint 元数据。

    Args:
        ckpt_info: 由 `get_checkpoint_info` 返回的字典。
    """
    if 'error' in ckpt_info:
        print(f"[WARNING] 无法读取 checkpoint 信息: {ckpt_info['error']}")
        return
    
    print("=" * 80)
    print(f"[INFO] Loaded Checkpoint: {ckpt_info['ckpt_name']}")
    print(f"  Epoch: {ckpt_info.get('epoch', 'N/A')}")
    print(f"  Global Step: {ckpt_info.get('global_step', 'N/A')}")
    
    hparams = ckpt_info.get('hparams', {})
    if hparams:
        print(f"  Model Type: MetMamba")
        print(f"  In Shape: {hparams.get('in_shape', 'N/A')}")
        print(f"  Out Seq Length: {hparams.get('out_seq_length', 'N/A')}")
    print("=" * 80)

# ==========================================
# Part 2: 核心统计计算 (Core Metrics)
# ==========================================
def calc_seq_metrics(true_seq: np.ndarray, pred_seq: np.ndarray, verbose: bool = True) -> dict:
    """计算序列预测的综合气象评分（调用 MetScore）。

    该函数将 Numpy 数据转换为 Tensor，使用官方评分模块 `MetScore` 计算
    包括相关系数 (Corr)、Threat Score (TS) 和 平均绝对误差 (MAE) 在内的综合得分。

    Args:
        true_seq (np.ndarray): 真实观测序列 (归一化值 0~1)。
            Shape: [T, H, W], 其中 T=20 (通常), H=256, W=256。
        pred_seq (np.ndarray): 模型预测序列 (归一化值 0~1)。
            Shape: [T, H, W]。
        verbose (bool): 是否打印逐帧详细评分日志。默认 True。

    Returns:
        dict: 包含以下键的字典:
            - "final_score" (float): 时间加权的序列总评分。
            - "score_per_frame" (np.ndarray): 每一帧的综合评分, Shape [T]。
            - "pred_clean" (np.ndarray): 去噪后的预测序列 (用于可视化), Shape [T, H, W]。
    """
    # 1. 初始化评分模块
    # MetScore 内部包含固定的评分权重和阈值
    MM_MAX = 30.0
    scorer = MetScore(data_max=MM_MAX)
    
    # 2. 数据转换 (Numpy -> Tensor)
    # MetScore 需要 [B, T, H, W] 格式
    # Shape: [T, H, W] -> [1, T, H, W]
    t_tensor = torch.from_numpy(true_seq).unsqueeze(0).float()
    p_tensor = torch.from_numpy(pred_seq).unsqueeze(0).float()
    
    # 3. 计算指标 (核心逻辑委托给 MetScore)
    # output keys: 'total_score', 'score_time', 'r_time', 'ts_time', 'mae_time'
    metrics = scorer(p_tensor, t_tensor)
    
    # 4. 提取结果并转回 Numpy
    final_score = metrics['total_score'].item()
    score_k_arr = metrics['score_time'].detach().numpy() # [T]
    r_k_arr = metrics['r_time'].detach().numpy()         # [T]
    
    # 获取时间权重用于日志显示 (从 buffer 中获取)
    # Shape: [T]
    time_weights = scorer.time_weights_default.detach().numpy()
    if len(time_weights) > len(score_k_arr):
        time_weights = time_weights[:len(score_k_arr)]

    # 5. 生成用于可视化的 Clean 数据
    # 模拟原始逻辑：将微小噪声置零 (MetScore 内部计算时不依赖此步骤，但在可视化时更清晰)

    pred_clean = pred_seq.copy()
    THRESHOLD_NOISE = 0.05
    denoise_mask = pred_clean < (THRESHOLD_NOISE / MM_MAX)
    pred_clean[denoise_mask] = 0.0

    # 6. 打印详细日志
    if verbose:
        # 反归一化用于统计信息打印
        tru_mm_seq = true_seq * 30.0
        prd_mm_seq = pred_clean * 30.0
        
        print(f"True Stats (mm): Max={np.max(tru_mm_seq):.2f}, Mean={np.mean(tru_mm_seq):.2f}")
        print(f"Pred Stats (mm): Max={np.max(prd_mm_seq):.2f}, Mean={np.mean(prd_mm_seq):.2f}")
        print("-" * 90)
        # 这里的 Weighted_Metric 是除去相关系数调节项后的分数部分，用于调试
        # Score = Term_Corr * Weighted_Metric => Weighted_Metric = Score / Term_Corr
        # Term_Corr = sqrt(exp(R - 1))
        print(f"{'T':<3} | {'Corr(R)':<9} | {'Score_base':<9} | {'Score_k':<9} | {'W_time':<8}")
        print("-" * 90)

        for t in range(len(score_k_arr)):
            R_k = r_k_arr[t]
            Score_k = score_k_arr[t]
            
            # 逆向计算 Base Score 用于展示 (Score_k / Corr_Term)
            term_corr = np.sqrt(np.exp(R_k - 1.0))
            base_score = Score_k / (term_corr + 1e-8)
            
            w_t = time_weights[t] if t < len(time_weights) else 0.0
            print(f"{t:<3} | {R_k:<9.4f} | {base_score:<9.4f} | {Score_k:<9.4f} | {w_t:<8.4f}")

    # 打印统计摘要
    print("-" * 90)
    # MetScore 返回的是 Tensor [L]
    ts_mean = metrics['ts_levels'].detach().numpy()
    mae_mean = metrics['mae_levels'].detach().numpy()
    
    ts_str = ", ".join([f"{v:.3f}" for v in ts_mean])
    mae_str = ", ".join([f"{v:.3f}" for v in mae_mean])
    
    print(f"[METRIC] TS_mean  (Levels): {ts_str}")
    print(f"[METRIC] MAE_mean (Levels): {mae_str}")
    print(f"[METRIC] Corr_mean: {np.mean(r_k_arr):.4f}")
    print(f"[METRIC] Final_Weighted_Score: {final_score:.6f}")
    print(f"[METRIC] Score_per_t: {', '.join([f'{s:.3f}' for s in score_k_arr])}")
    print("-" * 90)
    
    return {
        "final_score": final_score,
        "score_per_frame": score_k_arr,
        "pred_clean": pred_clean
    }

# ==========================================
# Part 3: 绘图功能 (Visualization)
# ==========================================

def create_precipitation_cmap():
    """创建自定义降水色标 (Colormap)。
    
    定义符合气象标准的降水色阶：
    区间定义: 
      - < 0.1: 白色 (无降水/微量)
      - 0.1 ~ 1.0: 浅绿
      - 1.0 ~ 2.0: 中绿
      - ...
      - > 8.0: 深红 (强降水)

    Returns:
        tuple: (cmap, norm)
            - cmap (matplotlib.colors.ListedColormap): 自定义颜色映射对象。
            - norm (matplotlib.colors.BoundaryNorm): 对应的边界归一化对象。
    """
    hex_colors = [
        # '#9CF48D',  # 0.01 <= r < 0.1 (浅绿 - 已注释)
        '#3CB73A',  # 0.1 <= r < 1 (中绿)
        '#63B7FF',  # 1 <= r < 2 (浅蓝)
        '#0200F9',  # 2 <= r < 5 (深蓝)
        '#EE00F0',  # 5 <= r < 8 (紫红)
        '#9F0000'   # r >= 8 (深红)
    ]
    
    cmap = mcolors.ListedColormap(hex_colors)
    cmap.set_bad('white')
    cmap.set_under('white')
    
    # 边界设置：起始值设为0.1，<0.1 落入 under 区域显示白色
    bounds = [0.1, 1, 2, 5, 8, 100]
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=len(hex_colors))
    
    return cmap, norm

def plot_seq_visualization(obs_seq: np.ndarray, 
                           true_seq: np.ndarray, 
                           pred_seq: np.ndarray, 
                           scores: np.ndarray, 
                           out_path: str, 
                           vmax: float = 1.0):
    """生成并保存多帧对比的可视化图像。

    图像包含 4 行：
    1. Obs: 输入的历史雷达/降水观测。
    2. GT: 未来的真实降水 (Target)。
    3. Pred: 模型的预测降水。
    4. Diff: 预测误差 (GT - Pred)。

    Args:
        obs_seq (np.ndarray): 输入序列, Shape [T_in, H, W] 或 [T_in, C, H, W]。
        true_seq (np.ndarray): 真实序列, Shape [T_out, H, W]。
        pred_seq (np.ndarray): 预测序列, Shape [T_out, H, W]。
        scores (np.ndarray): 每一帧的评分数组 (未使用，可用于标题扩展), Shape [T_out]。
        out_path (str): 图片保存路径。
        vmax (float): 归一化最大值 (默认 1.0)。
    """
    T = true_seq.shape[0] # T_out = 20
    rows, cols = 4, T
    
    precip_cmap, precip_norm = create_precipitation_cmap()
    
    # 反归一化到物理量 (mm)
    # MM_MAX 硬编码为 30.0，与训练一致
    MM_MAX = 30.0
    obs_mm = obs_seq * MM_MAX
    true_mm = true_seq * MM_MAX
    pred_mm = pred_seq * MM_MAX

    # 阈值过滤：小于 0.1mm 视为无降水 (可视化降噪)
    thr = 0.1
    obs_mm[obs_mm < thr] = 0
    true_mm[true_mm < thr] = 0
    pred_mm[pred_mm < thr] = 0
    
    # 增加高度以容纳底部 Legend
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5 + 1.0))
    if T == 1: axes = axes[:, np.newaxis]
    
    fig.subplots_adjust(bottom=0.12, top=0.95, left=0.02, right=0.98, wspace=0.1, hspace=0.3)

    # 辅助函数：统一设置边框样式
    def setup_ax_border(ax, show_ylabel=False, ylabel_text=""):
        """设置子图边框和标签。
        
        隐藏刻度但保留边框线，以便清晰显示图像边界。
        """
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 显式开启边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)
            
        if show_ylabel:
            ax.set_ylabel(ylabel_text, fontsize=8)
            # 确保 Y 轴 Label 显示
            ax.yaxis.label.set_visible(True)

    for t in range(T):
        # 1. Obs (Input)
        ax = axes[0, t]
        if t < obs_mm.shape[0]:
            ax.imshow(obs_mm[t], cmap=precip_cmap, norm=precip_norm)
            ax.set_title(f'In-{t}', fontsize=6)
        else:
            # 如果 Obs 长度小于 Out 长度，补黑/白
            ax.imshow(np.zeros_like(true_mm[0]), cmap=precip_cmap, norm=precip_norm)
        
        setup_ax_border(ax, show_ylabel=(t==0), ylabel_text='Obs')

        # 2. GT (Target)
        ax = axes[1, t]
        ax.imshow(true_mm[t], cmap=precip_cmap, norm=precip_norm)
        ax.set_title(f'T+{t+1}', fontsize=6)
        setup_ax_border(ax, show_ylabel=(t==0), ylabel_text='GT')

        # 3. Pred
        ax = axes[2, t]
        ax.imshow(pred_mm[t], cmap=precip_cmap, norm=precip_norm)
        setup_ax_border(ax, show_ylabel=(t==0), ylabel_text='Pred')
        
        # 4. Diff (GT - Pred)
        ax = axes[3, t]
        diff = true_mm[t] - pred_mm[t]
        # 差值图使用红蓝配色 (bwr)，范围固定为 [-30, 30] mm
        ax.imshow(diff, cmap='bwr', vmin=-30, vmax=30)
        setup_ax_border(ax, show_ylabel=(t==0), ylabel_text='Diff')

    # --- Legend 1: Precipitation (左侧) ---
    cbar_ax_precip = fig.add_axes([0.20, 0.05, 0.25, 0.015])
    sm_precip = plt.cm.ScalarMappable(cmap=precip_cmap, norm=precip_norm)
    sm_precip.set_array([])
    cbar_p = fig.colorbar(sm_precip, cax=cbar_ax_precip, orientation='horizontal', spacing='uniform')
    cbar_p.set_ticks([0.1, 1, 2, 5, 8])
    cbar_p.set_ticklabels(['0.1', '1', '2', '5', '8'])
    cbar_p.set_label('Precipitation (mm/6min)', fontsize=8)
    cbar_p.ax.tick_params(labelsize=7)
    
    # --- Legend 2: Diff (右侧) ---
    cbar_ax_diff = fig.add_axes([0.55, 0.05, 0.25, 0.015])
    sm_diff = plt.cm.ScalarMappable(cmap='bwr', norm=plt.Normalize(vmin=-30, vmax=30))
    sm_diff.set_array([])
    cbar_d = fig.colorbar(sm_diff, cax=cbar_ax_diff, orientation='horizontal')
    cbar_d.set_ticks([-30, -15, 0, 15, 30])
    cbar_d.set_label('Difference (mm)', fontsize=8)
    cbar_d.ax.tick_params(labelsize=7)

    print(f'[INFO] Saving Plot to {out_path}')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

# ==========================================
# Part 4: 主入口函数 (Wrapper)
# ==========================================
def render(obs_seq: torch.Tensor, 
           true_seq: torch.Tensor, 
           pred_seq: torch.Tensor, 
           out_path: str, 
           vmax: float = 1.0) -> float:
    """渲染单个样本的预测结果，计算指标并绘图。

    Args:
        obs_seq (torch.Tensor): 观测张量 (输入).
            Shape: [C, H, W] 或 [T, C, H, W] (C=0 为雷达/降水).
        true_seq (torch.Tensor): 真实值张量. Shape: [T, 1, H, W] 或 [T, H, W].
        pred_seq (torch.Tensor): 预测值张量. Shape: [T, 1, H, W] 或 [T, H, W].
        out_path (str): 图片输出路径.
        vmax (float): 归一化最大值.

    Returns:
        float: 该样本的最终加权综合评分 (Final Score).
    """
    # 1. 数据格式统一 (转 Numpy & 提取主要通道)
    def to_numpy_ch(x, ch=0):
        if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
        # 输入维度 (T, C, H, W), 通道 0 是雷达/降水标签
        if x.ndim == 4: x = x[:, ch] 
        return x

    obs = to_numpy_ch(obs_seq) # 取 ch=0 (Radar/Label)
    tru = to_numpy_ch(true_seq)
    prd = to_numpy_ch(pred_seq)
    
    print(f"Processing: {os.path.basename(out_path)}")
    
    # 2. 调用统计模块 (已重构为使用 MetScore)
    metrics_res = calc_seq_metrics(tru, prd, verbose=True)
    
    final_score = metrics_res['final_score']
    
    # 3. 调用绘图模块
    plot_seq_visualization(obs, tru, metrics_res['pred_clean'], metrics_res['score_per_frame'], out_path, vmax=vmax)
    
    return final_score

def parse_args():
    parser = argparse.ArgumentParser(description='Test SCWDS Mamba Model')
    parser.add_argument('--data_path', type=str, default='data/samples.jsonl')
    parser.add_argument('--in_shape', type=int, nargs=4, default=[10, 54, 256, 256],
                        help='Input shape (T, C, H, W). Default: [10, 54, 256, 256]')
    parser.add_argument('--out_seq_length', type=int, default=20,
                        help='Output sequence length. Default: 20')
    parser.add_argument('--save_dir', type=str, default='./output/mamba')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--accelerator', type=str, default='cuda')
    parser.add_argument('--ckpt_path', type=str, default=None, 
                        help='Path to specific checkpoint file. If None, auto-search in save_dir.')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.accelerator if torch.cuda.is_available() else 'cpu')
    
    # 1. Config & Paths
    if args.ckpt_path and os.path.exists(args.ckpt_path):
        ckpt_path = args.ckpt_path
        print(f"[INFO] Using specified checkpoint: {ckpt_path}")
    else:
        if args.ckpt_path:
            print(f"[WARN] Specified checkpoint not found: {args.ckpt_path}. Falling back to search.")
        ckpt_path = find_best_ckpt(args.save_dir)
    
    ckpt_info = get_checkpoint_info(ckpt_path)
    epoch = ckpt_info.get('epoch', None)
    
    if epoch is None:
        epoch = 0
    
    out_dir = os.path.join(args.save_dir, f'vis_{epoch:02d}')
    os.makedirs(out_dir, exist_ok=True)
    
    log_file_path = os.path.join(out_dir, 'log.txt')
    set_logger(log_file_path)
    
    print(f"[INFO] Starting Test on {device}")
    print_checkpoint_info(ckpt_info)
    print(f"[INFO] Input Shape: {args.in_shape}")
    print(f"[INFO] Output Length: {args.out_seq_length}")
    print(f"[INFO] Metric MM_MAX: 30.0") 
    print(f"[INFO] 可视化结果将保存到: {out_dir}")
    
    # 加载 MetMamba 模型
    model = MetMambaTrainer.load_from_checkpoint(ckpt_path)
    model.eval().to(device)
    
    # 2. Data
    data_module = ScwdsDataModule(
        data_path=args.data_path,
        batch_size=1,
        num_workers=4
    )
    data_module.setup('test')
    test_loader = data_module.test_dataloader()
    
    scores = []
    
    with torch.no_grad():
        for bidx, batch in enumerate(test_loader):
            metadata_batch, batch_x, batch_y, input_mask, target_mask = batch
            
            # 调用 test_step (传入 tuple)
            outputs = model.test_step(
                (None, batch_x.to(device), batch_y.to(device), None, target_mask.to(device), ), 
                bidx
            )
            
            save_path = os.path.join(out_dir, f'sample_{bidx:03d}.png')
            s = render(outputs['inputs'], outputs['trues'], outputs['preds'], save_path)
            scores.append(s)
            
            if bidx >= args.num_samples - 1:
                break
    
    if len(scores) > 0:
        print(f"\n[FINAL] Average Score ({len(scores)} samples): {np.mean(scores):.6f}")
    
    restore_stdout()

if __name__ == '__main__':
    try:
        main()
    finally:
        restore_stdout()