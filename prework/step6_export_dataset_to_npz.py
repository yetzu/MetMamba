# prework/step6_export_dataset_to_npz.py
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import multiprocessing
import time

# 确保能导入项目根目录下的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metai.dataset import ScwdsDataset

# --- 全局变量 ---
worker_ds = None 
global_output_dir = None
global_target_size = None
EXPECTED_FILE_SIZE = 148112142

def resize_tensor(data_np: np.ndarray, target_shape: tuple, mode: str = 'max_pool') -> np.ndarray:
    if data_np is None: return None
    tensor = torch.from_numpy(data_np).unsqueeze(0)
    B, T, C, H, W = tensor.shape
    target_H, target_W = target_shape
    if H == target_H and W == target_W: return data_np
    
    is_bool = tensor.dtype == torch.bool
    if is_bool: tensor = tensor.float()
    tensor = tensor.view(B * T, C, H, W)
    
    if mode == 'max_pool':
        if target_H < H or target_W < W:
            processed = F.adaptive_max_pool2d(tensor, output_size=target_shape)
        else:
            processed = F.interpolate(tensor, size=target_shape, mode='bilinear', align_corners=False)
    elif mode in ['nearest', 'bilinear']:
        align = False if mode == 'bilinear' else None
        processed = F.interpolate(tensor, size=target_shape, mode=mode, align_corners=align)
    else:
        raise ValueError(f"Unsupported interpolation mode: {mode}")
    
    processed = processed.view(B, T, C, target_H, target_W)
    if is_bool: processed = processed.bool()
    return processed.squeeze(0).numpy()

def worker_init(data_path, output_dir, target_size):
    global worker_ds, global_output_dir, global_target_size
    torch.set_num_threads(1)
    try:
        worker_ds = ScwdsDataset(data_path=data_path, is_train=True)
    except Exception as e:
        print(f"[Worker Error] Dataset init failed: {e}")
        worker_ds = None
    global_output_dir = output_dir
    global_target_size = target_size

def process_sample(idx):
    # 子进程只负责生成，不负责检查（主进程已过滤）
    global worker_ds, global_output_dir, global_target_size
    
    if worker_ds is None: return False, f"Dataset not initialized"

    try:
        batch = worker_ds[idx]
        metadata, input_data, target_data, input_mask, target_mask = batch
        
        sample_id = metadata.get('sample_id', f'sample_{idx}')
        save_path = os.path.join(global_output_dir, f"{sample_id}.npz")
        
        input_data_resized = resize_tensor(input_data, global_target_size, mode='max_pool')
        target_data_resized = resize_tensor(target_data, global_target_size, mode='max_pool')
        target_mask_resized = resize_tensor(target_mask, global_target_size, mode='nearest')
        
        save_dict = {
            "input_data": input_data_resized,
            "target_data": target_data_resized,
            "target_mask": target_mask_resized
        }
        
        np.savez(save_path, **save_dict)
        return True, None
        
    except Exception as e:
        sid = 'Unknown'
        try: sid = worker_ds.samples[idx].get('sample_id', 'Unknown')
        except: pass
        return False, f"Sample {idx} ({sid}) failed: {e}"

def get_valid_existing_files(output_dir):
    """
    快速扫描输出目录。
    优化点：先检查文件名（快），再检查 is_file 和 size（慢）。
    """
    valid_ids = set()
    if not os.path.exists(output_dir):
        return valid_ids
    
    print(f"[Step 2] 正在扫描输出目录 (标准大小: {EXPECTED_FILE_SIZE})...")
    print(f"         目录: {output_dir}")
    
    # 估算文件数量用于进度条（如果目录巨大，这一步本身可能略慢，但通常很快）
    # 如果感觉 scandir 卡住，通常是因为在遍历网络文件系统
    try:
        # 使用 scandir 迭代器，配合 tqdm
        with os.scandir(output_dir) as entries:
            # mininterval=0.5 防止刷新太快拖慢 IO
            for entry in tqdm(entries, desc="Scanning Files", unit="file", mininterval=0.5):
                # 优化：先判断后缀名（字符串操作，极快），不符合直接跳过，避免系统调用
                if entry.name.endswith('.npz'): 
                    # 再判断是否是文件（稍微慢一点点，通常 cached）
                    if entry.is_file():
                        # 最后判断大小（产生 stat 系统调用，最慢）
                        # 使用 follow_symlinks=False 进一步减少开销
                        if entry.stat(follow_symlinks=False).st_size == EXPECTED_FILE_SIZE:
                            sample_id = entry.name[:-4]
                            valid_ids.add(sample_id)
    except Exception as e:
        print(f"[WARNING] 扫描目录出错: {e}")

    print(f"         发现 {len(valid_ids)} 个有效文件。")
    return valid_ids

def export_train_data_parallel(data_path, output_dir, target_size=(256, 256), num_workers=8):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 1. 加载 Dataset ---
    print("\n" + "="*50)
    print("[Step 1] 加载样本列表 (JSONL)...")
    start_time = time.time()
    # ScwdsDataset 读取大文件可能需要几秒到几十秒，这里虽然无法给内部加进度条
    # 但明确的 print 能让你知道它没死机
    temp_ds = ScwdsDataset(data_path=data_path, is_train=True)
    all_samples = temp_ds.samples
    total_samples = len(temp_ds)
    del temp_ds 
    print(f"         加载完成。共 {total_samples} 个样本 (耗时 {time.time()-start_time:.2f}s)")

    # --- 2. 扫描已完成文件 ---
    print("-" * 30)
    valid_ids = get_valid_existing_files(output_dir)
    
    # --- 3. 生成任务列表 ---
    print("-" * 30)
    print("[Step 3] 对比差异，生成任务队列...")
    todo_indices = []
    skipped_count = 0
    
    # 使用 tqdm 显示对比进度（虽然通常很快，但量大时也需要）
    for idx, sample in enumerate(tqdm(all_samples, desc="Comparing")):
        sid = sample.get('sample_id')
        if sid in valid_ids:
            skipped_count += 1
        else:
            todo_indices.append(idx)
            
    print(f"\n[任务统计]")
    print(f"  总数   : {total_samples}")
    print(f"  已完成 : {skipped_count} (跳过)")
    print(f"  待处理 : {len(todo_indices)}")
    
    if len(todo_indices) == 0:
        print("\n[DONE] 所有文件均已存在且完整，无需处理。")
        return

    # --- 4. 并行处理 ---
    print("-" * 30)
    print(f"[Step 4] 启动并行处理 (Workers: {num_workers})...")
    
    pool = multiprocessing.Pool(
        processes=num_workers,
        initializer=worker_init,
        initargs=(data_path, output_dir, target_size)
    )

    success_count = 0
    fail_count = 0
    
    try:
        results = list(tqdm(
            pool.imap_unordered(process_sample, todo_indices, chunksize=5),
            total=len(todo_indices),
            desc="Exporting",
            smoothing=0.05 # 降低平滑度，让速度显示更灵敏
        ))
    except KeyboardInterrupt:
        print("\n[STOP] 用户中断处理。正在关闭进程池...")
        pool.terminate()
        pool.join()
        sys.exit(1)

    pool.close()
    pool.join()

    # 5. 统计
    for success, msg in results:
        if success:
            success_count += 1
        else:
            fail_count += 1
            if msg: print(f"\n[WARNING] {msg}")

    print(f"\n" + "="*50)
    print(f"[DONE] 处理结束。")
    print(f"本次生成: {success_count}")
    print(f"此前已存: {skipped_count}")
    print(f"失败报错: {fail_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="并行导出 ScwdsDataset 为 .npz")
    parser.add_argument("--data_path", type=str, default="data/samples.jsonl", help="样本索引路径")
    parser.add_argument("--output_dir", type=str, default="/data/zjobs/SevereWeather_AI_2025/CP/Train", help="保存目录")
    parser.add_argument("--num_workers", type=int, default=50, help="并行进程数")
    args = parser.parse_args()
    
    export_train_data_parallel(
        data_path=args.data_path,
        output_dir=args.output_dir,
        target_size=(256, 256),
        num_workers=args.num_workers
    )