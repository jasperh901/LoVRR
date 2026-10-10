import argparse
import json

import yaml

from lovrr.inference.pipeline import LoVRR


def main():
    parser = argparse.ArgumentParser(description="LoVRR video reasoning")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--video", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--problem-type", default=None)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    runtime = LoVRR.from_config(config)
    result = runtime.answer(args.video, args.question, problem_type=args.problem_type)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
