"""
lit_gpt.py

LightningModule + GPT-2 模型定义
--------------------------------
这个文件的职责只有两个：
1. 决定 GPT-2 模型的“体型”（tiny / small 等） -> build_gpt2_config
2. 把 GPT-2 包装成一个 PyTorch LightningModule -> LitGPT2

整体思路：
- 模型结构和前向传播交给 HuggingFace 的 GPT2LMHeadModel
- 训练流程（train/val step、optimizer、scheduler）交给 Lightning 的 Trainer
"""

from typing import Any, Dict, Optional

import torch
from torch.optim import AdamW
import pytorch_lightning as pl
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


def build_gpt2_config(
    model_size: str,
    vocab_size: int,
    n_positions: int = 256,
) -> GPT2Config:
    """
    根据 model_size 构造 GPT-2 的配置。

    - "small": 类 GPT-2 small (~117M 参数)，作为主实验模型
    - "tiny" : 缩小版 (~10M 级别)，用于本地调试 / sanity check

    参数说明:
    - model_size : 字符串，决定用哪一种体型配置（"small" 或 "tiny"）
    - vocab_size : tokenizer 词表大小
    - n_positions: 最大位置编码长度（即 context 窗口，用于 n_ctx / n_positions）
    """
    if model_size == "small":
        # GPT-2 small 标准配置：12 层，768 hidden，12 heads
        return GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_ctx=n_positions,
            n_embd=768,
            n_layer=12,
            n_head=12,
        )
    elif model_size == "tiny":
        # 迷你版 GPT-2：4 层，256 hidden，4 heads
        # 方便在 Mac / CPU 上快速调试 pipeline
        return GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_ctx=n_positions,
            n_embd=256,
            n_layer=4,
            n_head=4,
        )
    else:
        raise ValueError(f"Unknown model_size: {model_size}")


class LitGPT2(pl.LightningModule):
    """
    LightningModule 封装 GPT-2 训练逻辑。

    这个类要完成几件事：
    - __init__           : 初始化 tokenizer + GPT-2 模型 + 保存超参数
    - forward            : 调用底层 GPT2LMHeadModel 的前向
    - training_step      : 定义一批数据的训练 loss 计算和 logging
    - validation_step    : 定义验证集上 loss 的计算和 logging
    - configure_optimizers: 定义优化器和学习率调度器

    简单理解：
    - data_module.py 决定“喂什么数据”
    - LitGPT2 决定“脑子长什么样 + 怎么学习”
    """

    def __init__(
        self,
        model_size: str,
        tokenizer_name: str,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        total_steps: Optional[int] = None,
    ):
        """
        参数说明:
        - model_size     : "tiny" 或 "small"，对应不同 GPT-2 规模
        - tokenizer_name : HF tokenizer 名称，例如 "gpt2"
        - learning_rate  : 初始学习率
        - weight_decay   : 权重衰减系数 (AdamW)
        - warmup_steps   : scheduler 的 warmup 步数
        - total_steps    : 总训练步数，用于 scheduler；如果为 None，则只返回 optimizer
        """
        super().__init__()

        # Lightning 自动把 __init__ 的参数记录到 self.hparams 里，
        # 方便 checkpoint 保存 / 恢复 / logging。
        self.save_hyperparameters()

        # 保险：确保这些超参数类型正确（YAML 解析有时会给出 str）
        self.hparams.learning_rate = float(learning_rate)
        self.hparams.weight_decay = float(weight_decay)
        self.hparams.warmup_steps = int(warmup_steps)

        # 1) 初始化 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # GPT-2 没有 pad_token，通常把 eos 当作 pad_token 使用
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2) 根据 model_size 构建 GPT-2 配置
        config = build_gpt2_config(
            model_size=model_size,
            vocab_size=self.tokenizer.vocab_size,
            n_positions=256,  # Tiny 实验用 256，正式实验可以用 512/1024
        )

        # 3) 根据配置初始化 GPT-2 语言模型（从头训练，不加载预训练权重）
        self.model = GPT2LMHeadModel(config)

        # 4) 用于 scheduler 的 total_steps（可在外部估计好后传进来）
        self.total_steps = total_steps

    # ----- 前向传播 -----
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        标准 forward，直接调用 GPT2LMHeadModel 的 forward。

        输入:
        - input_ids     : [batch, seq_len]
        - attention_mask: [batch, seq_len] (可选)
        - labels        : [batch, seq_len] (可选，给出则会返回 cross-entropy loss)
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    # ----- 训练步骤 -----
    def training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        单个训练 step 的逻辑。

        batch 由 DataCollatorForLanguageModeling 生成，包含:
        - input_ids
        - attention_mask
        - labels  (即 next-token 预测的目标)

        Lightning 会使用返回的 loss 自动做:
        - backward()
        - optimizer.step()
        - zero_grad()
        """

        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss

        # logging:
        # - on_step=True  : 每个 step 打一次
        # - on_epoch=True : 每个 epoch 聚合一个平均值
        # - prog_bar=True : 在进度条里显示
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # ----- 验证步骤 -----
    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        验证集上的单个 step 逻辑。

        和 training_step 类似，但不做反向传播，只计算 loss 并 log。
        """
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        val_loss = outputs.loss
        # 只在 epoch 级别聚合显示 val_loss
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)
        return val_loss

    # ----- 优化器 & 学习率调度 -----
    def configure_optimizers(self):
        """
        定义优化器和（可选）学习率调度器。

        采用标准 Transformer 训练配置：
        - AdamW
        - 对 bias / LayerNorm.weight 不做 weight decay
        - 线性 warmup + 线性 decay (get_linear_schedule_with_warmup)
        """

        # 这些参数名中含有 "bias" 或 "LayerNorm.weight" 的不做 weight decay
        no_decay = ["bias", "LayerNorm.weight"]

        # 强制转成 float，避免 YAML / Lightning 把它们变成字符串
        lr: float = float(self.hparams.learning_rate)
        wd: float = float(self.hparams.weight_decay)

        # 将参数拆成两组：带 weight decay 和不带 weight decay
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": wd,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        # AdamW 是 Transformer 训练的标配优化器
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=lr,
        )

        # 如果没有提供 total_steps，就只返回 optimizer，不设置 scheduler
        if self.total_steps is None:
            return optimizer

        # 使用线性 warmup + 线性衰减的学习率调度器
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.hparams.warmup_steps),
            num_training_steps=int(self.total_steps),
        )

        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "step",  # 每个 step 更新一次 lr
            "frequency": 1,
        }

        # Lightning 期望返回 ([optimizers], [schedulers])
        return [optimizer], [scheduler_dict]

    # 可选：如果你想在 datamodule setup 后再设置 total_steps，可以扩展这个方法
    def set_total_steps(self, total_steps: int) -> None:
        """
        外部辅助函数：如果你在 Trainer / DataModule 中重新估算了 total_steps，
        可以通过这个函数在训练开始前更新它，用于更准确的 scheduler 配置。
        """
        self.total_steps = int(total_steps)

