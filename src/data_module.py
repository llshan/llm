from typing import Optional, Dict, Any
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class WikiLMDataModule(pl.LightningDataModule):
    """
    LightningDataModule for causal LM training on Wikipedia-like text.

    假设数据是 jsonl，每行包含一个字段: "text"
    """

    def __init__(
        self,
        tokenizer_name: str,
        train_path: str,
        val_path: str,
        block_size: int = 128,
        train_batch_size: int = 2,
        val_batch_size: int = 2,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.dataset: Optional[DatasetDict] = None
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is not None:
            return

        raw_datasets = load_dataset(
            "json",
            data_files={"train": self.hparams.train_path,
                        "validation": self.hparams.val_path},
        )

        def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
            return self.tokenizer(
                examples["text"],
                truncation=False,
            )

        tokenized = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Tokenizing dataset",
        )

        block_size = self.hparams.block_size

        def group_texts(examples: Dict[str, Any]) -> Dict[str, Any]:
            concatenated = []
            for ids in examples["input_ids"]:
                concatenated.extend(ids)

            total_length = (len(concatenated) // block_size) * block_size
            if total_length == 0:
                return {"input_ids": [], "attention_mask": []}

            input_ids = [
                concatenated[i: i + block_size]
                for i in range(0, total_length, block_size)
            ]
            attention_mask = [[1] * block_size for _ in range(len(input_ids))]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        lm_datasets = tokenized.map(
            group_texts,
            batched=True,
            desc=f"Grouping texts into blocks of size {block_size}",
        )

        self.dataset = lm_datasets

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset["train"],
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self.data_collator,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.data_collator,
        )

