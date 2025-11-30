#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
export_wiki_raw.py

从 HuggingFace 的 `wikipedia` 数据集导出一个英文子集，存成 jsonl：
每行包含: {"id": ..., "title": ..., "text": ...}

用途：
- 为 preprocess_wiki.py 提供原始输入 (raw_wiki_train/val.jsonl)
- 后续再生成 original / de-entity / shuffled 三个变体

用法示例：
    uv run python export_wiki_raw.py \
        --version 20220301.en \
        --output_dir data \
        --train_articles 200000 \
        --val_articles 5000 \
        --min_chars 200
"""

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


def export_wiki_raw(
    config: str,
    output_dir: str,
    train_articles: int = 200_000,
    val_articles: int = 5_000,
    min_chars: int = 200,
) -> None:
    """
    从 HuggingFace `wikipedia` 数据集导出 train/val jsonl。

    参数:
    - version        : wikipedia 数据集的版本字符串，例如 "20220301.en"
    - output_dir     : 输出目录
    - train_articles : 训练集导出多少篇
    - val_articles   : 验证集导出多少篇（从 train 后面截取）
    - min_chars      : 文本长度少于该阈值的样本会被过滤
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Loading wikimedia/wikipedia config={config} from HuggingFace...")
    ds = load_dataset("wikimedia/wikipedia", config, split="train")
    total = len(ds)
    print(f"[INFO] Total articles in split 'train': {total}")

    # 我们简单地从前面顺序取 train_articles + val_articles 篇，
    # 中间按 min_chars 做过滤。
    train_path = Path(output_dir) / "raw_wiki_train.jsonl"
    val_path = Path(output_dir) / "raw_wiki_val.jsonl"

    train_f = open(train_path, "w", encoding="utf-8")
    val_f = open(val_path, "w", encoding="utf-8")

    try:
        train_count = 0
        val_count = 0

        for i, ex in enumerate(ds):
            if train_count >= train_articles and val_count >= val_articles:
                break

            text = ex.get("text", "")
            if not text or len(text) < min_chars:
                continue

            obj = {
                "id": ex.get("id", str(i)),
                "title": ex.get("title", ""),
                "text": text,
            }

            if train_count < train_articles:
                train_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                train_count += 1
            elif val_count < val_articles:
                val_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                val_count += 1

            if (train_count + val_count) % 5000 == 0:
                print(f"[INFO] Collected train={train_count}, val={val_count} samples...")

        print(f"[DONE] Final counts: train={train_count}, val={val_count}")
        print(f"[DONE] Train file: {train_path}")
        print(f"[DONE] Val   file: {val_path}")

    finally:
        train_f.close()
        val_f.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Export raw Wikipedia jsonl from HF `wikipedia` dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="20220301.en",
        help="wikimedia/wikipedia dataset config, e.g. '20231101.en' (language=en).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory to save raw_wiki_train.jsonl and raw_wiki_val.jsonl",
    )
    parser.add_argument(
        "--train_articles",
        type=int,
        default=200_000,
        help="Number of training articles to export.",
    )
    parser.add_argument(
        "--val_articles",
        type=int,
        default=5_000,
        help="Number of validation articles to export.",
    )
    parser.add_argument(
        "--min_chars",
        type=int,
        default=200,
        help="Skip articles shorter than this number of characters.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    export_wiki_raw(
        config=args.config,
        output_dir=args.output_dir,
        train_articles=args.train_articles,
        val_articles=args.val_articles,
        min_chars=args.min_chars,
    )


if __name__ == "__main__":
    main()

