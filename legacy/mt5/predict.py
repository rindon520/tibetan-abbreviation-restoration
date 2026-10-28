"""
藏文古籍缩写词还原 - 推理脚本
用法:
    # 单句预测
    python predict.py --text "缩写文本"
    
    # 批量预测文件
    python predict.py --input_file test_source.txt --output_file results.txt
    
    # 交互模式
    python predict.py --interactive
"""

import argparse
import logging
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TibetanRestorer:
    def __init__(
        self,
        model_dir: str = "final_model",
        device: str = None,
        num_beams: int = 4,
        max_target_length: int = 256,
        source_prefix: str = "藏文缩写还原: ",
    ):
        self.source_prefix = source_prefix
        self.num_beams = num_beams
        self.max_target_length = max_target_length

        # 自动选择设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"使用设备: {self.device}")
        logger.info(f"加载模型: {model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("模型加载完成")

    @torch.no_grad()
    def predict(self, texts: list[str]) -> list[str]:
        """批量预测（列表输入，列表输出）"""
        texts_with_prefix = [self.source_prefix + t for t in texts]

        inputs = self.tokenizer(
            texts_with_prefix,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_length=self.max_target_length,
            num_beams=self.num_beams,
            early_stopping=True,
            no_repeat_ngram_size=0,   # 藏文不限重复ngram
        )

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [d.strip() for d in decoded]

    def predict_one(self, text: str) -> str:
        """单句预测"""
        return self.predict([text])[0]

    def evaluate_file(self, source_file: str, restore_file: str, batch_size: int = 32):
        """评估文件并输出准确率"""
        with open(source_file, "r", encoding="utf-8") as f:
            sources = [l.strip() for l in f if l.strip()]
        with open(restore_file, "r", encoding="utf-8") as f:
            targets = [l.strip() for l in f if l.strip()]

        assert len(sources) == len(targets)
        logger.info(f"评估样本数: {len(sources)}")

        predictions = []
        for i in range(0, len(sources), batch_size):
            batch = sources[i:i + batch_size]
            preds = self.predict(batch)
            predictions.extend(preds)
            if (i // batch_size) % 10 == 0:
                logger.info(f"进度: {i}/{len(sources)}")

        # 计算指标
        exact_match = sum(p == t for p, t in zip(predictions, targets)) / len(targets)

        def char_sim(pred, label):
            if not label:
                return 1.0 if not pred else 0.0
            matches = sum(p == l for p, l in zip(pred, label))
            return matches / max(len(pred), len(label))

        char_acc = sum(char_sim(p, t) for p, t in zip(predictions, targets)) / len(targets)

        print(f"\n{'='*40}")
        print(f"评估结果 ({len(sources)} 条)")
        print(f"  完全匹配率: {exact_match*100:.2f}%")
        print(f"  字符准确率: {char_acc*100:.2f}%")
        print(f"{'='*40}\n")

        # 输出错误样本示例
        errors = [
            (s, t, p)
            for s, t, p in zip(sources, targets, predictions)
            if p != t
        ]
        print(f"错误样本示例（共 {len(errors)} 条）:")
        for s, t, p in errors[:10]:
            print(f"  输入:   {s}")
            print(f"  正确:   {t}")
            print(f"  预测:   {p}")
            print()

        return predictions


def main():
    parser = argparse.ArgumentParser(description="藏文缩写词还原推理")
    parser.add_argument("--model_dir", type=str, default="final_model")
    parser.add_argument("--text", type=str, help="单句缩写文本")
    parser.add_argument("--input_file", type=str, help="批量输入文件（每行一条）")
    parser.add_argument("--output_file", type=str, help="批量输出文件")
    parser.add_argument("--eval_source", type=str, help="评估用缩写文件")
    parser.add_argument("--eval_restore", type=str, help="评估用还原文件")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    restorer = TibetanRestorer(
        model_dir=args.model_dir,
        num_beams=args.num_beams,
    )

    # 单句预测
    if args.text:
        result = restorer.predict_one(args.text)
        print(f"输入: {args.text}")
        print(f"还原: {result}")

    # 批量文件预测
    elif args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            sources = [l.strip() for l in f if l.strip()]

        predictions = []
        for i in range(0, len(sources), args.batch_size):
            batch = sources[i:i + args.batch_size]
            preds = restorer.predict(batch)
            predictions.extend(preds)
            logger.info(f"进度: {min(i+args.batch_size, len(sources))}/{len(sources)}")

        output_file = args.output_file or "predictions.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(pred + "\n")
        logger.info(f"预测结果已保存至: {output_file}")

    # 评估模式
    elif args.eval_source and args.eval_restore:
        restorer.evaluate_file(args.eval_source, args.eval_restore, args.batch_size)

    # 交互模式
    elif args.interactive:
        print("藏文缩写词还原（输入 'quit' 退出）")
        print("-" * 40)
        while True:
            text = input("请输入缩写文本: ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            result = restorer.predict_one(text)
            print(f"还原结果: {result}\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
