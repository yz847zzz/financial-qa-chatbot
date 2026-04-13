"""
vLLM server configuration.

Single source of truth for adapter names, server URL, and generation defaults.
Used by deployment/api/client.py and can be imported by start_server.sh via
`python -c "from deployment.server.vllm_config import ..."`
"""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


# ── Adapter registry ──────────────────────────────────────────────────────────
# Maps logical role → vLLM model name as registered in --lora-modules.
# "base" is the served-model-name for the base model (no adapter).
#
# SGMV batching:  when two requests in the same batch use different adapters
#                 (e.g. one "nl2sql" and one "base"), vLLM automatically uses
#                 the SGMV kernel to apply each adapter's delta weights to the
#                 correct sequences — no serialization overhead between adapters.
ADAPTERS = {
    "base":     "base",      # plain Llama — intent, decompose, answer, rewrite
    "nl2sql":   "nl2sql",    # NL→SQL LoRA adapter
    "intent":   "base",      # intent classifier (LoRA not yet trained → base)
    "rewriter": "base",      # query rewriter   (LoRA not yet trained → base)
}

# When a LoRA adapter is trained for intent/rewriter, flip "base" → adapter name:
#   ADAPTERS["intent"]   = "intent"
#   ADAPTERS["rewriter"] = "rewriter"


@dataclass
class VLLMConfig:
    host: str = "localhost"
    port: int = 8001

    # Generation defaults per role — override per-call if needed
    max_tokens: dict = field(default_factory=lambda: {
        "intent":    5,     # single label output
        "decompose": 300,   # JSON list of sub-questions
        "rewrite":   200,   # JSON list of rewritten queries
        "nl2sql":    200,   # SQL SELECT statement
        "answer":    500,   # final answer
        "direct":    300,   # Type3 chat response
        "synthesize":600,   # combined answer
    })

    temperature: float = 0.0   # greedy decoding everywhere (SQL must be deterministic)
    timeout_s:   float = 60.0  # per-request HTTP timeout

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def adapter_for(self, role: str) -> str:
        """Return the vLLM model name for a given role."""
        return ADAPTERS.get(role, "base")

    def max_new_tokens(self, role: str) -> int:
        return self.max_tokens.get(role, 256)


# Default singleton — import this in client.py
DEFAULT_CONFIG = VLLMConfig()
