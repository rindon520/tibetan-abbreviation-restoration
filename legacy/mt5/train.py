"""
藏文古籍缩写词还原模型 - 修复版训练脚本 v3
修复:
  1. 移除废弃的 as_target_tokenizer()，修复 loss=nan 问题
  2. 改用 Seq2SeqTrainer，训练时实时显示准确率
  3. 网络超时处理（禁用safetensors自动转换）
  4. tokenizer → processing_class（兼容 transformers 4.46+）
  5. 移除废弃的 logging_dir / warmup_ratio 警告
GPU: RTX 5080 16GB
"""

import os
import random
import logging
import warnings
import numpy as np

# ===== 必须在导入transformers之前设置 =====
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SAFETENSORS_CONVERSION"] = "1"  # 修复SSL超时
# 网络慢时取消注释：
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from datasets import Dataset
from transformers import (
    MT5ForConditionalGeneration,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
#  配置区
# ============================================================
class Config:
    SOURCE_FILE     = "data/Source.txt"
    RESTORE_FILE    = "data/Restore.txt"
    MODEL_NAME      = "google/mt5-base"
    OUTPUT_DIR      = "./checkpoints"
    FINAL_MODEL_DIR = "./final_model"
    TASK_PREFIX         = "藏文缩写还原: "
    MAX_SOURCE_LENGTH   = 128
    MAX_TARGET_LENGTH   = 256
    TRAIN_RATIO     = 0.85
    VAL_RATIO       = 0.10
    EPOCHS          = 30
    BATCH_SIZE      = 16
    GRAD_ACCUM      = 4           # 等效 batch=64
    LEARNING_RATE   = 3e-4
    WEIGHT_DECAY    = 0.01
    WARMUP_RATIO    = 0.1
    BF16            = True        # RTX 5080 支持 bf16
    FP16            = False
    EVAL_STEPS      = 200
    SAVE_STEPS      = 200
    LOGGING_STEPS   = 50
    NUM_BEAMS       = 4
    SEED            = 42
    PATIENCE        = 5


# ============================================================
#  数据加载
# ============================================================
def load_raw_data(source_file, restore_file):
    with open(source_file, "r", encoding="utf-8") as f:
        sources = [l.strip() for l in f]
    with open(restore_file, "r", encoding="utf-8") as f:
        restores = [l.strip() for l in f]
    pairs = [(s, r) for s, r in zip(sources, restores) if s and r]
    logger.info(f"有效数据对: {len(pairs)} 条")
    return [p[0] for p in pairs], [p[1] for p in pairs]


def split_data(sources, restores, train_ratio, val_ratio, seed):
    random.seed(seed)
    idx = list(range(len(sources)))
    random.shuffle(idx)
    n = len(idx)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    def sel(i_list):
        return [sources[i] for i in i_list], [restores[i] for i in i_list]
    train = sel(idx[:n_train])
    val   = sel(idx[n_train: n_train + n_val])
    test  = sel(idx[n_train + n_val:])
    logger.info(f"划分 -> 训练:{len(train[0])} | 验证:{len(val[0])} | 测试:{len(test[0])}")
    return train, val, test


# ============================================================
#  数据预处理（关键修复：不使用 as_target_tokenizer）
# ============================================================
def make_preprocess_fn(tokenizer, cfg):
    def preprocess(examples):
        inputs = [cfg.TASK_PREFIX + s for s in examples["source"]]
        # ✅ 修复：直接tokenize输入
        model_inputs = tokenizer(
            inputs,
            max_length=cfg.MAX_SOURCE_LENGTH,
            truncation=True,
            padding=False,
        )
        # ✅ 修复：用 text_target 参数编码标签（替代废弃的 as_target_tokenizer）
        labels = tokenizer(
            text_target=examples["restore"],
            max_length=cfg.MAX_TARGET_LENGTH,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return preprocess


# ============================================================
#  评估指标（训练时实时显示）
# ============================================================
def make_compute_metrics(tokenizer, val_sources=None):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # ✅ 修复：predictions 可能含 -100 或超出词表的无效ID，统一替换为 pad_token_id
        predictions = np.where(
            (predictions >= 0) & (predictions < tokenizer.vocab_size),
            predictions,
            tokenizer.pad_token_id,
        ).astype(np.int64)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels_fixed = np.where(labels != -100, labels, tokenizer.pad_token_id).astype(np.int64)
        decoded_labels = tokenizer.batch_decode(labels_fixed, skip_special_tokens=True)

        preds = [p.strip() for p in decoded_preds]
        refs  = [r.strip() for r in decoded_labels]

        # 精确匹配率
        exact_match = sum(p == r for p, r in zip(preds, refs)) / len(preds)

        # 字符级准确率
        total_chars, correct_chars = 0, 0
        for p, r in zip(preds, refs):
            n = max(len(p), len(r))
            total_chars += n
            correct_chars += sum(a == b for a, b in zip(p, r))
        char_acc = correct_chars / total_chars if total_chars > 0 else 0.0

        # ROUGE-L
        def lcs_len(s1, s2):
            dp = [0] * (len(s2) + 1)
            for c in s1:
                prev = 0
                new_dp = [0] * (len(s2) + 1)
                for j, d in enumerate(s2):
                    new_dp[j+1] = prev + 1 if c == d else max(dp[j+1], new_dp[j])
                    prev = dp[j+1]
                dp = new_dp
            return dp[len(s2)]

        rouge_scores = []
        for p, r in zip(preds, refs):
            if not p or not r:
                rouge_scores.append(0.0); continue
            l = lcs_len(p, r)
            prec = l / len(p); rec = l / len(r)
            f1 = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0.0
            rouge_scores.append(f1)
        rouge_l = sum(rouge_scores) / len(rouge_scores)

        # 打印样例
        logger.info("\n===== 验证样例 =====")
        for i in range(min(4, len(preds))):
            flag = "✓" if preds[i] == refs[i] else "✗"
            src_text = val_sources[i] if val_sources else ""
            logger.info(f"  [{flag}] 输入: {src_text}")
            logger.info(f"       预测: {preds[i]}")
            logger.info(f"       真实: {refs[i]}")

        return {
            "exact_match":   round(exact_match, 4),
            "char_accuracy": round(char_acc, 4),
            "rouge_l":       round(rouge_l, 4),
        }
    return compute_metrics


# ============================================================
#  主函数
# ============================================================
def main():
    cfg = Config()
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)

    if torch.cuda.is_available():
        gpu  = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu} | 显存: {vram:.1f}GB")
    else:
        logger.warning("未检测到GPU，使用CPU训练")

    # 1. 加载数据
    logger.info("=" * 60)
    sources, restores = load_raw_data(cfg.SOURCE_FILE, cfg.RESTORE_FILE)
    logger.info("数据预览:")
    for i in range(min(3, len(sources))):
        logger.info(f"  缩写: {sources[i]}  ->  还原: {restores[i]}")

    (train_src, train_tgt), (val_src, val_tgt), (test_src, test_tgt) = split_data(
        sources, restores, cfg.TRAIN_RATIO, cfg.VAL_RATIO, cfg.SEED
    )

    train_ds = Dataset.from_dict({"source": train_src, "restore": train_tgt})
    val_ds   = Dataset.from_dict({"source": val_src,   "restore": val_tgt})
    test_ds  = Dataset.from_dict({"source": test_src,  "restore": test_tgt})

    # 2. 加载模型
    logger.info("=" * 60)
    logger.info(f"加载模型: {cfg.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, use_fast=False)
    model = MT5ForConditionalGeneration.from_pretrained(cfg.MODEL_NAME)
    logger.info(f"参数量: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

    # 3. 预处理
    logger.info("=" * 60)
    logger.info("预处理数据集...")
    fn = make_preprocess_fn(tokenizer, cfg)
    train_ds = train_ds.map(fn, batched=True, remove_columns=["source","restore"], desc="训练集")
    val_ds   = val_ds.map(fn, batched=True, remove_columns=["source","restore"], desc="验证集")
    test_ds  = test_ds.map(fn, batched=True, remove_columns=["source","restore"], desc="测试集")

    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
    )

    # 4. 训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.OUTPUT_DIR,
        num_train_epochs=cfg.EPOCHS,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        per_device_eval_batch_size=cfg.BATCH_SIZE * 2,
        gradient_accumulation_steps=cfg.GRAD_ACCUM,
        learning_rate=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
        warmup_ratio=cfg.WARMUP_RATIO,
        max_grad_norm=1.0,
        bf16=cfg.BF16,
        fp16=cfg.FP16,
        eval_strategy="steps",
        eval_steps=cfg.EVAL_STEPS,
        save_strategy="steps",
        save_steps=cfg.SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="rouge_l",
        greater_is_better=True,
        logging_steps=cfg.LOGGING_STEPS,
        report_to="none",
        predict_with_generate=True,
        generation_max_length=cfg.MAX_TARGET_LENGTH,
        generation_num_beams=cfg.NUM_BEAMS,
        dataloader_num_workers=0,   # Windows必须为0
        dataloader_pin_memory=True,
        seed=cfg.SEED,
    )

    # 5. 训练
    logger.info("=" * 60)
    logger.info("开始训练！训练时每隔200步自动评估并显示准确率")
    logger.info(f"实际 batch_size = {cfg.BATCH_SIZE} × {cfg.GRAD_ACCUM} = {cfg.BATCH_SIZE * cfg.GRAD_ACCUM}")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,   # ✅ 修复：新版transformers(4.46+)改名为processing_class
        data_collator=collator,
        compute_metrics=make_compute_metrics(tokenizer, val_src),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.PATIENCE)],
    )

    trainer.train()

    # 6. 保存
    logger.info("=" * 60)
    logger.info(f"保存最佳模型 -> {cfg.FINAL_MODEL_DIR}")
    trainer.save_model(cfg.FINAL_MODEL_DIR)
    tokenizer.save_pretrained(cfg.FINAL_MODEL_DIR)

    # 7. 测试集评估
    logger.info("=" * 60)
    logger.info("测试集最终评估...")
    results = trainer.predict(test_ds)
    logger.info("===== 测试集结果 =====")
    for k, v in results.metrics.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    with open("test_results.txt", "w", encoding="utf-8") as f:
        for k, v in results.metrics.items():
            f.write(f"{k}: {v}\n")

    logger.info("完成！结果已保存至 test_results.txt")


if __name__ == "__main__":
    main()
