# metai/model/mamba/trainer.py

import torch
import torch.nn.functional as F
import lightning as l
from typing import Any, Dict, cast
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from metai.model.core import get_optim_scheduler
from .model import MetMamba
from .loss import HybridLoss

class MetMambaTrainer(l.LightningModule):
    """
    MetMamba Lightning Trainer
    """
    def __init__(self, **args):
        super().__init__()
        self.save_hyperparameters()
        config: Dict[str, Any] = dict(args)
        
        # 1. 构建 MetMamba 模型
        self.model = self._build_model(config)
        
        # 2. 初始化混合 Loss
        self.criterion = HybridLoss(
            l1_weight=config.get('loss_weight_l1', 1.0),
            ssim_weight=config.get('loss_weight_ssim', 0.5),
            csi_weight=config.get('loss_weight_csi', 1.0),
            spectral_weight=config.get('loss_weight_spectral', 0.1),
            evo_weight=config.get('loss_weight_evo', 0.5)
        )
        
        rs = config.get('resize_shape', None)
        self.resize_shape = tuple(rs) if rs is not None else None
        self.use_curriculum = config.get('use_curriculum_learning', True)

    def _build_model(self, config):
        return MetMamba(
            in_shape=config.get('in_shape'),
            hid_S=config.get('hid_S', 64),
            hid_T=config.get('hid_T', 256),
            N_S=config.get('N_S', 4),
            N_T=config.get('N_T', 8),
            mlp_ratio=config.get('mlp_ratio', 4.0),
            drop=config.get('drop', 0.0),
            drop_path=config.get('drop_path', 0.0),
            spatio_kernel_enc=config.get('spatio_kernel_enc', 3),
            spatio_kernel_dec=config.get('spatio_kernel_dec', 3),
            out_channels=config.get('out_channels', 1),
            out_seq_length=config.get('out_seq_length', 20),
            d_state=config.get('d_state', 16),
            d_conv=config.get('d_conv', 4),
            expand=config.get('expand', 2),
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer, scheduler, by_epoch = get_optim_scheduler(
            self.hparams, self.hparams.get('max_epochs', 100), self.model
        )
        return cast(OptimizerLRScheduler, {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch" if by_epoch else "step"},
        })
    
    def on_train_epoch_start(self):
        """Curriculum Learning: Dynamic Loss Weights"""
        if not self.use_curriculum: return
        
        progress = self.current_epoch / self.hparams.get('max_epochs', 100)
        
        # 动态权重策略
        weights = {
            'l1': max(10.0 - (9.0 * (progress ** 0.5)), 1.0),
            'ssim': 1.0 - 0.5 * progress,
            'csi': 0.5 + 4.5 * (progress ** 2),
            'spec': 0.1 * progress,
            'evo': 0.5 * progress
        }
        
        if hasattr(self.criterion, 'weights'):
            self.criterion.weights.update(weights)
        
        for k, v in weights.items():
            self.log(f"train/w_{k}", v, on_epoch=True, sync_dist=True)

    
    def forward(self, x):
        return self.model(x)
    
    def _interp(self, tensor, mode='max_pool'):
        if self.resize_shape is None: return tensor
        B, T, C, H, W = tensor.shape
        target_H, target_W = self.resize_shape
        if H == target_H and W == target_W: return tensor

        flat = tensor.view(B * T, C, H, W)
        if mode == 'max_pool':
            out = F.adaptive_max_pool2d(flat, (target_H, target_W))
        else:
            out = F.interpolate(flat, size=(target_H, target_W), mode=mode)
        return out.view(B, T, C, target_H, target_W)
    
    def training_step(self, batch, batch_idx):
        _, x, y, _, mask = batch

        x = self._interp(x, 'max_pool')
        y = self._interp(y, 'max_pool')
        mask = mask.bool()
        mask = self._interp(mask.float(), 'nearest')

        pred = self(x)
        loss, loss_dict = self.criterion(pred, y, mask=mask)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        _, x, y, _, mask = batch

        x = self._interp(x, 'max_pool')
        y = self._interp(y, 'max_pool')
        mask = mask.bool()
        mask = self._interp(mask.float(), 'nearest')
        
        logits_pred = self(x)
        y_pred = torch.sigmoid(logits_pred)
        y_pred_clamped = torch.clamp(y_pred, 0.0, 1.0)
        
        # 2. 计算 Loss (保持不变)
        loss, _ = self.criterion(logits_pred, y, mask=mask)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)

        # ====================================================
        # 3. [优化] 对齐官方规则的评分计算
        # ====================================================
        MM_MAX = 30.0
        pred_mm = y_pred_clamped * MM_MAX
        target_mm = y * MM_MAX

        # A. 官方阈值与强度权重 (Table 2)
        # 去掉了 0.01 (噪音)，对齐官方 0.1 起步
        thresholds = [0.1, 1.0, 2.0, 5.0, 8.0]
        level_weights = [0.1, 0.1, 0.2, 0.25, 0.35]
        
        # B. 官方时效权重 (Table 1) - 针对 20 帧
        # 对应 6min 到 120min
        time_weights_list = [
            0.0075, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,  # 1-10 (60min 权重最高)
            0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.0075, 0.005 # 11-20
        ]
        # 转换为 Tensor 并移动到对应设备
        T_out = pred_mm.shape[1]
        if T_out == 20:
            time_weights = torch.tensor(time_weights_list, device=self.device)
        else:
            # 如果输出不是20帧，则平均分配
            time_weights = torch.ones(T_out, device=self.device) / T_out

        # C. 计算加权 TS (Weighted TS)
        # 这种计算方式是 "Micro-average over Batch, but Macro over Time/Level"
        # 既保留了批量计算的速度，又引入了时效权重
        
        total_score = 0.0
        total_level_weight = sum(level_weights)

        for t_val, w_level in zip(thresholds, level_weights):
            # [B, T, H, W] (C=1, 已去除) -> Bool
            # 注意：target_mm 和 pred_mm 可能是 [B, T, 1, H, W]，需要squeeze或指定sum dim
            # 在training step中，logits是 [B, T, C, H, W]，squeeze后 [B, T, H, W]
            # 让我们兼容这两种情况
            
            # 确保是 [B, T, H, W]
            if pred_mm.dim() == 5 and pred_mm.shape[2] == 1:
                 p_mm = pred_mm.squeeze(2)
                 t_mm = target_mm.squeeze(2)
            else:
                 p_mm = pred_mm
                 t_mm = target_mm

            hits_tensor = (p_mm >= t_val) & (t_mm >= t_val)
            misses_tensor = (p_mm < t_val) & (t_mm >= t_val)
            false_alarms_tensor = (p_mm >= t_val) & (t_mm < t_val)
            
            # 在 [B, H, W] 维度求和，保留 [T] 维度以应用时效权重
            # sum dim: 0(Batch), 2(H), 3(W) -> Result shape: [T]
            # [FIX]: 输入已经是 4D [B, T, H, W]，所以 dim=(0, 2, 3) 是正确的
            # 如果输入是 5D [B, T, C, H, W]，则需要 dim=(0, 2, 3, 4)
            
            if p_mm.dim() == 4:
                sum_dims = (0, 2, 3)
            else: # 5D
                sum_dims = (0, 2, 3, 4)

            hits = hits_tensor.float().sum(dim=sum_dims)
            misses = misses_tensor.float().sum(dim=sum_dims)
            false_alarms = false_alarms_tensor.float().sum(dim=sum_dims)
            
            # 计算每帧的 TS: [T]
            ts_t = hits / (hits + misses + false_alarms + 1e-6)
            
            # 应用时效权重: sum( [T] * [T] ) -> Scalar
            ts_weighted_time = (ts_t * time_weights).sum()
            
            # 累加强度分级得分
            total_score += ts_weighted_time * w_level

        # 归一化 (虽然 level_weights 和为 1，但保持严谨)
        val_score = total_score / total_level_weight

        # 4. 记录指标
        self.log('val_score', val_score, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # 额外记录 MAE 供参考 (不参与 EarlyStopping，因为 MAE 容易被 0 值主导)
        val_mae = F.l1_loss(y_pred_clamped, y)
        self.log('val_mae', val_mae, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        _, x, y, _, mask = batch

        x = self._interp(x, 'max_pool')
        y = self._interp(y, 'max_pool')
        mask = mask.bool()
        mask = self._interp(mask.float(), 'nearest')

        logits_pred = self(x)
        y_pred = torch.sigmoid(logits_pred)
        y_pred_clamped = torch.clamp(y_pred, 0.0, 1.0)
        
        with torch.no_grad():
            loss, loss_dict = self.criterion(logits_pred, y, mask=mask)
            
        self.log('test_loss', loss, on_epoch=True)
        
        return {
            'inputs': x[0].cpu().float().numpy(),
            'preds': y_pred_clamped[0].cpu().float().numpy(),
            'trues': y[0].cpu().float().numpy()
        }
    
    def infer_step(self, batch, batch_idx):
        metadata, x, input_mask = batch 
        x = self._interp(x, mode='max_pool')
        logits_pred = self(x)
        y_pred = torch.sigmoid(logits_pred)
        return torch.clamp(y_pred, 0.0, 1.0)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.infer_step(batch, batch_idx)