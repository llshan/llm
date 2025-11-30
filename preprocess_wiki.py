#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
preprocess_wiki.py

将原始 Wikipedia jsonl（raw_wiki_*.jsonl）预处理为 LM 训练可用的三种变体：

1) original  : 仅做基本清洗（空白折叠等）
2) deentity  : 命名实体脱敏（优先用 spaCy NER，无则用简易规则）
3) shuffled  : 打乱结构（按 token 或句子）

输入格式假设为 jsonl，每行至少包含字段:
    {"id": "...", "title": "...", "text": "Full article text ..."}

输出格式为 jsonl，每行仅包含:
    {"text": "..."}   # 已处理后的文本

用法示例：
    uv run python preprocess_wiki.py \
      --input data/raw_wiki_train.jsonl \
      --output_dir data/processed_train \
      --variants original deentity shuffled \
      --min_chars 200 \
      --use_spacy_ner \
      --shuffle_mode tokens
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# 尝试导入 spaCy（可选）
try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    SPACY_AVAILABLE = False


# ========== 基础工具 ==========

def iter_jsonl(path: str) -> Iterable[Dict]:
    """逐行读取 jsonl，yield 每行的 dict。"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # 坏行直接跳过
                continue


def normalize_whitespace(text: str) -> str:
    """基础清洗：去掉多余换行/空白，折叠为一个空格。"""
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ========== 变体 1：original ==========

def process_original(text: str) -> str:
    """原始版：只做空白清洗，不动内容。"""
    return normalize_whitespace(text)


# ========== 变体 2：deentity（实体脱敏） ==========

# 常见实体类型 → 占位符映射
ENTITY_LABEL_MAP = {
    "PERSON": "[ENT_PERSON]",
    "ORG": "[ENT_ORG]",
    "GPE": "[ENT_GPE]",
    "LOC": "[ENT_LOC]",
    "NORP": "[ENT_GROUP]",
    "FAC": "[ENT_FACILITY]",
    # 其它实体统一归为 MISC
}


def load_spacy_model(model_name: str = "en_core_web_sm"):
    """加载 spaCy NER 模型。"""
    if not SPACY_AVAILABLE:
        raise ImportError(
            "spaCy 未安装。请先运行：\n"
            "  uv add spacy\n"
            "  uv run python -m spacy download en_core_web_sm\n"
        )
    return spacy.load(model_name)


def deentity_with_spacy(text: str, nlp) -> str:
    """
    使用 spaCy NER 做实体脱敏：
    - PERSON/ORG/GPE/... → 统一占位符，如 [ENT_PERSON]
    - 非实体 token 保留
    """
    text = normalize_whitespace(text)
    doc = nlp(text)

    result_tokens: List[str] = []
    i = 0
    while i < len(doc):
        token = doc[i]
        ent_type = token.ent_type_  # 实体类型（如 PERSON, ORG 等）
        ent_iob = token.ent_iob_    # B / I / O

        if ent_type and ent_iob == "B":
            placeholder = ENTITY_LABEL_MAP.get(ent_type, "[ENT_MISC]")
            result_tokens.append(placeholder)

            # 跳过整个实体 span
            j = i + 1
            while j < len(doc) and doc[j].ent_iob_ == "I":
                j += 1
            i = j
        elif ent_type and ent_iob == "I":
            # 理论上不会走到这里（已在 B span 中跳过），保险起见
            i += 1
        else:
            result_tokens.append(token.text)
            i += 1

    return " ".join(result_tokens)


# 简单规则版：把“首字母大写的词”粗暴当作实体
CAPITALIZED_WORD_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")


def deentity_with_simple_rule(text: str) -> str:
    """
    简易规则脱敏（不依赖 spaCy）：
    - 粗糙地把“首字母大写的词”替换为 [ENT_CAP]
    - 可作为 baseline 或无 spaCy 环境下的退路
    """
    text = normalize_whitespace(text)

    def repl(match):
        word = match.group(0)
        # 一些常见功能词可以直接放行
        if word in {"The", "This", "That", "A", "An"}:
            return word
        return "[ENT_CAP]"

    return CAPITALIZED_WORD_RE.sub(repl, text)


# ========== 变体 3：shuffled（打乱） ==========

def shuffle_tokens(text: str, seed: Optional[int] = None) -> str:
    """
    token 级打乱：
    - 按空格切分为 token
    - 全局打乱顺序
    - 再拼回一句话
    完全破坏句法，但保留 token 频率。
    """
    text = normalize_whitespace(text)
    tokens = text.split(" ")
    if len(tokens) <= 1:
        return text
    if seed is not None:
        rnd = random.Random(seed)
        rnd.shuffle(tokens)
    else:
        random.shuffle(tokens)
    return " ".join(tokens)


# 简单句子切分：按 .!? + 空格 切
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def shuffle_sentences(text: str, seed: Optional[int] = None) -> str:
    """
    句子级打乱：
    - 粗略用正则按 .!? 分句
    - 打乱句子顺序
    - 再拼回文章
    句内语法保留，文章结构被打乱。
    """
    text = normalize_whitespace(text)
    sentences = SENT_SPLIT_RE.split(text)
    sentences = [s for s in sentences if s.strip()]
    if len(sentences) <= 1:
        return text

    if seed is not None:
        rnd = random.Random(seed)
        rnd.shuffle(sentences)
    else:
        random.shuffle(sentences)
    return " ".join(sentences)


# ========== 主处理逻辑 ==========

def process_wiki(
    input_path: str,
    output_dir: str,
    variants: List[str],
    max_articles: Optional[int] = None,
    min_chars: int = 200,
    use_spacy_ner: bool = False,
    shuffle_mode: str = "tokens",  # "tokens" or "sentences"
    seed: int = 42,
) -> None:
    """
    从 raw Wikipedia jsonl 生成多个变体 jsonl 文件。

    参数:
    - input_path    : 原始 jsonl 路径（raw_wiki_xxx.jsonl）
    - output_dir    : 输出目录
    - variants      : 要生成的变体列表，如 ["original", "deentity", "shuffled"]
    - max_articles  : 最多处理多少篇（None 表示全部）
    - min_chars     : 文本长度少于该阈值的样本会被过滤
    - use_spacy_ner : deentity 是否使用 spaCy NER（否则用简单规则）
    - shuffle_mode  : "tokens"（打乱 token）或 "sentences"（打乱句子）
    - seed          : 随机种子（影响 shuffled）
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    out_files: Dict[str, any] = {}

    try:
        # 为每个变体准备输出文件
        for v in variants:
            out_path = Path(output_dir) / f"wiki_{v}.jsonl"
            out_files[v] = open(out_path, "w", encoding="utf-8")
            print(f"[INFO] Writing variant '{v}' to: {out_path}")

        # 如需 spaCy NER，预先加载
        nlp = None
        if "deentity" in variants and use_spacy_ner:
            print("[INFO] Loading spaCy NER model 'en_core_web_sm' ...")
            nlp = load_spacy_model("en_core_web_sm")

        count = 0
        for article in iter_jsonl(input_path):
            if max_articles is not None and count >= max_articles:
                break

            text = article.get("text", "")
            if not text or len(text) < min_chars:
                continue

            # ----- original -----
            if "original" in variants:
                out_text = process_original(text)
                out_obj = {"text": out_text}
                out_files["original"].write(json.dumps(out_obj, ensure_ascii=False) + "\n")

            # ----- de-entity -----
            if "deentity" in variants:
                if use_spacy_ner and nlp is not None:
                    de_text = deentity_with_spacy(text, nlp)
                else:
                    de_text = deentity_with_simple_rule(text)
                out_obj = {"text": de_text}
                out_files["deentity"].write(json.dumps(out_obj, ensure_ascii=False) + "\n")

            # ----- shuffled -----
            if "shuffled" in variants:
                if shuffle_mode == "tokens":
                    sh_text = shuffle_tokens(text)
                else:
                    sh_text = shuffle_sentences(text)
                out_obj = {"text": sh_text}
                out_files["shuffled"].write(json.dumps(out_obj, ensure_ascii=False) + "\n")

            count += 1
            if count % 1000 == 0:
                print(f"[INFO] Processed {count} articles...")

        print(f"[DONE] Total processed articles: {count}")

    finally:
        for f in out_files.values():
            try:
                f.close()
            except Exception:
                pass


# ========== CLI ==========

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess raw Wikipedia jsonl into LM-ready variants."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to raw Wikipedia jsonl file (e.g. data/raw_wiki_train.jsonl).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save processed wiki_*.jsonl files.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["original", "deentity", "shuffled"],
        choices=["original", "deentity", "shuffled"],
        help="Which variants to generate.",
    )
    parser.add_argument(
        "--max_articles",
        type=int,
        default=None,
        help="Maximum number of articles to process (default: all).",
    )
    parser.add_argument(
        "--min_chars",
        type=int,
        default=200,
        help="Skip articles shorter than this number of characters.",
    )
    parser.add_argument(
        "--use_spacy_ner",
        action="store_true",
        help="Use spaCy NER for de-entity. If not set, use simple regex rule.",
    )
    parser.add_argument(
        "--shuffle_mode",
        type=str,
        default="tokens",
        choices=["tokens", "sentences"],
        help="How to shuffle for 'shuffled' variant: tokens or sentences.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    process_wiki(
        input_path=args.input,
        output_dir=args.output_dir,
        variants=args.variants,
        max_articles=args.max_articles,
        min_chars=args.min_chars,
        use_spacy_ner=args.use_spacy_ner,
        shuffle_mode=args.shuffle_mode,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

