import argparse
import sys
from pathlib import Path

from train_restore import generate_text, load_model_bundle


def configure_utf8_stdout(stream=sys.stdout):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Restore Tibetan abbreviations.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/tibetan_restore/best.pt"))
    parser.add_argument("--text", default="")
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-len", type=int, default=128)
    return parser.parse_args()


def main():
    configure_utf8_stdout()
    args = parse_args()
    device = None if args.device == "auto" else args.device
    model, tokenizer, _, device = load_model_bundle(args.checkpoint, device=device)

    if args.input_file:
        lines = args.input_file.read_text(encoding="utf-8").splitlines()
    elif args.text:
        lines = [args.text]
    else:
        lines = [line.rstrip("\n") for line in sys.stdin]

    outputs = [
        generate_text(model, tokenizer, line, device, max_len=args.max_len) for line in lines
    ]

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text("\n".join(outputs) + "\n", encoding="utf-8")
    else:
        for output in outputs:
            print(output)


if __name__ == "__main__":
    main()
