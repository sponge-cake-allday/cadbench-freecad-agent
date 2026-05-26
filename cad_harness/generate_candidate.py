import argparse
from pathlib import Path

from .llm_client import chat_completion, extract_python_code
from .models import CadTask


SYSTEM_PROMPT = """You are a CAD generation agent.
Generate a single self-contained FreeCAD Python script.
The script must:
- import FreeCAD as App and Part when needed
- create the requested model using millimeter dimensions
- save the final document to the path in CAD_HARNESS_OUTPUT_FCSTD when that env var exists
- call doc.recompute() before saving
- print the output path

Return only Python code, with no explanation.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a FreeCAD candidate script with an OpenAI-compatible model.")
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="model")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    task = CadTask.from_path(args.task)
    content = chat_completion(
        base_url=args.base_url,
        model=args.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task.prompt_text()},
        ],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    code = extract_python_code(content)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(code, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
