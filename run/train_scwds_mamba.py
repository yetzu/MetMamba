# run/train_scwds_mamba.py
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightning.pytorch.cli import LightningCLI

# 导入你的 DataModule
from metai.dataset.met_dataloader_scwds import ScwdsDataModule


from metai.model.mamba import MetMambaTrainer as MeteoMambaModule 

def main():
    # 设置矩阵乘法精度，提升性能
    torch.set_float32_matmul_precision('high')

    # LightningCLI 自动处理：
    # 1. 解析命令行参数 (--model.xxx, --data.xxx, --trainer.xxx)
    # 2. 实例化 Model 和 DataModule
    # 3. 配置 Trainer
    # 4. 运行 trainer.fit() (因为 run=True)
    cli = LightningCLI(
        model_class=MeteoMambaModule,
        datamodule_class=ScwdsDataModule,
        save_config_callback=None,
        run=True,
        parser_kwargs={"parser_mode": "omegaconf"} 
    )

if __name__ == "__main__":
    main()