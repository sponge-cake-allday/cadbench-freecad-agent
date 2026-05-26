from harbor_agents.freecad_cad_agent import (
    CadBenchFreeCadAgent,
    SYSTEM_PROMPT,
    build_repair_prompt,
    sanitize_code,
)


class QwenCadAgent(CadBenchFreeCadAgent):
    """Backward-compatible import path for earlier Qwen-specific runs."""

    def __init__(self, *args, model_name: str | None = "qwen-coder", **kwargs):
        super().__init__(*args, model_name=model_name, **kwargs)

    @staticmethod
    def name() -> str:
        return "qwen-cad-agent"
