"""
vLLM OpenAI-compatible client — drop-in replacement for chatbot.py local inference.

This module mirrors every function in chatbot.py that is marked [VLLM_SWAP].
To switch chatbot.py from local Llama to vLLM:

    1. Start the server:
         bash deployment/scripts/start_server.sh          (inside WSL2)

    2. In chatbot.py, replace the local call section:
         from deployment.api.client import VLLMClient
         client = VLLMClient()
         # then call client.generate() instead of llm_generate()

    Or use the convenience wrappers below that exactly match the chatbot.py
    function signatures — just swap the imports.

Multi-adapter batching (SGMV):
    Each call specifies a `role` which maps to a vLLM model name (adapter).
    vLLM automatically batches requests for different adapters in the same GPU
    step using SGMV (Segmented Gather Matrix-Vector) kernels — effectively
    applying adapter A to some sequences and adapter B to others in one pass,
    with no serialization cost between them.

    Adapter → role mapping (defined in vllm_config.py):
        role="base"     → base Llama  (intent, decompose, rewrite, answer)
        role="nl2sql"   → NL2SQL LoRA (generate_sql only)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openai import OpenAI, APIConnectionError
from deployment.server.vllm_config import DEFAULT_CONFIG, VLLMConfig


class VLLMClient:
    """
    Thin wrapper around the OpenAI-compatible vLLM REST API.

    Usage:
        client = VLLMClient()            # uses DEFAULT_CONFIG (localhost:8001)
        client = VLLMClient(port=8002)   # custom port

        text = client.generate(
            messages=[{"role": "user", "content": "Hello"}],
            role="base",
        )
    """

    def __init__(self, config: VLLMConfig | None = None, **overrides):
        self.cfg = config or DEFAULT_CONFIG
        # Apply any keyword overrides (e.g. VLLMClient(port=8002))
        for k, v in overrides.items():
            setattr(self.cfg, k, v)

        self._openai = OpenAI(
            base_url=self.cfg.base_url,
            api_key="none",           # vLLM doesn't require an API key
            timeout=self.cfg.timeout_s,
        )

    def health(self) -> bool:
        """Return True if the server is reachable."""
        try:
            self._openai.models.list()
            return True
        except APIConnectionError:
            return False

    def generate(
        self,
        messages: list[dict],
        role: str = "base",
        max_new_tokens: int | None = None,
    ) -> str:
        """
        Send a chat-format prompt to vLLM and return the completion text.

        Args:
            messages:       list of {"role": ..., "content": ...} dicts
            role:           logical role → selects which adapter to use
                            ("base", "nl2sql", "intent", "rewriter")
            max_new_tokens: override the default for this role

        Returns:
            The model's reply as a plain string (stripped).
        """
        model  = self.cfg.adapter_for(role)
        n_tok  = max_new_tokens or self.cfg.max_new_tokens(role)

        resp = self._openai.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=n_tok,
            temperature=self.cfg.temperature,
        )
        return resp.choices[0].message.content.strip()

    # ── Convenience wrappers matching chatbot.py signatures ───────────────────
    # These let you swap `llm_generate(model, tok, messages, n)` → `client.*`
    # without changing any of the call sites.

    def llm_generate(self, model, tokenizer, messages: list[dict],
                     max_new_tokens: int = 256) -> str:
        """
        Drop-in for chatbot.llm_generate().
        `model` and `tokenizer` args are ignored (kept for signature compat).
        Uses role="base" (plain Llama, no adapter).
        """
        return self.generate(messages, role="base", max_new_tokens=max_new_tokens)

    def generate_sql_vllm(self, messages: list[dict],
                          max_new_tokens: int = 200) -> str:
        """
        Drop-in for chatbot.generate_sql() — routes to the nl2sql adapter.
        The NL2SQL LoRA is applied only to this call; other concurrent requests
        in the same batch continue using their own adapter via SGMV.
        """
        return self.generate(messages, role="nl2sql", max_new_tokens=max_new_tokens)


# ── Module-level convenience functions ────────────────────────────────────────
# These can be imported directly into chatbot.py as drop-in replacements:
#
#   from deployment.api.client import llm_generate, generate_sql_vllm
#
# Then in chatbot.py replace:
#   llm_generate(model, tokenizer, messages, n)
# with:
#   llm_generate(None, None, messages, n)       ← model/tok ignored
#
# And:
#   generate_sql(question, nl2sql_model, nl2sql_tok)
# becomes:
#   generate_sql_vllm(question)

_default_client: VLLMClient | None = None


def _get_client() -> VLLMClient:
    global _default_client
    if _default_client is None:
        _default_client = VLLMClient()
    return _default_client


def llm_generate(model, tokenizer, messages: list[dict],
                 max_new_tokens: int = 256) -> str:
    """Module-level drop-in for chatbot.llm_generate()."""
    return _get_client().generate(messages, role="base",
                                  max_new_tokens=max_new_tokens)


def generate_sql_vllm(messages: list[dict], max_new_tokens: int = 200) -> str:
    """Module-level drop-in for SQL generation — uses nl2sql adapter."""
    return _get_client().generate(messages, role="nl2sql",
                                  max_new_tokens=max_new_tokens)


# ── Health check entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    client = VLLMClient()
    print(f"Server: {client.cfg.base_url}")

    if not client.health():
        print("ERROR: vLLM server not reachable.")
        print("  Start it with: bash deployment/scripts/start_server.sh")
        sys.exit(1)

    print("Server is up. Testing adapters...\n")

    # Test base model
    resp = client.generate(
        messages=[{"role": "user", "content": "What is net income?"}],
        role="base",
        max_new_tokens=50,
    )
    print(f"[base]   → {resp}\n")

    # Test NL2SQL adapter
    from chatbot import NL2SQL_SYSTEM
    resp = client.generate_sql_vllm(
        messages=[
            {"role": "system",  "content": NL2SQL_SYSTEM},
            {"role": "user",    "content": "What was Apple's revenue in FY2023?"},
        ]
    )
    print(f"[nl2sql] → {resp}\n")

    print("All adapters responding correctly.")
