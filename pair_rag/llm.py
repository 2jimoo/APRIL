"""LLM backends shared by the attacker, the RAG generator and RAGForensics.

A model is given as "<provider>:<model>" (default: claude:$ANTHROPIC_MODEL from .env, else ollama:qwen2.5:7b):
    claude:claude-opus-5-5     Anthropic API (official `anthropic` SDK)
    ollama:qwen2.5:7b          local Ollama server (OLLAMA_HOST, default http://localhost:11434)
    openai:gpt-4o-mini         OpenAI-compatible API (only kept so the original RAGForensics setup still runs)
"""
import logging
import os
import time

log = logging.getLogger(__name__)

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_dotenv():
    """Read KEY=VALUE lines from .env in the project or its parent; real environment variables win."""
    for path in (os.path.join(_PROJECT, ".env"), os.path.join(os.path.dirname(_PROJECT), ".env")):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.removeprefix("export ").split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()
# Default for every LLM role: claude:$ANTHROPIC_MODEL when .env/environment names one, else local Ollama.
DEFAULT_LLM = f"claude:{os.environ['ANTHROPIC_MODEL']}" if os.environ.get("ANTHROPIC_MODEL") else "ollama:qwen2.5:7b"

# Models that accept output_config.effort / server-side refusal fallbacks.
# Haiku 4.5 rejects effort; Claude Haiku 5.5 accepts it (default medium) but has no server-side fallback.
_CLAUDE_NO_EFFORT_PREFIXES = ("claude-haiku-4", "claude-3")
_CLAUDE_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}


def parse_spec(spec):
    if ":" not in spec:
        raise ValueError(f"LLM spec must look like 'claude:<model>' or 'ollama:<model>', got {spec!r}")
    provider, model = spec.split(":", 1)
    provider = provider.lower()
    if provider not in ("claude", "anthropic", "ollama", "openai"):
        raise ValueError(f"Unknown provider {provider!r} in {spec!r}")
    return ("claude" if provider == "anthropic" else provider), model


def spec_to_name(spec):
    """File-name-safe model name used in attack_feedback paths (e.g. qwen2.5:7b -> qwen2.5-7b)."""
    _, model = parse_spec(spec)
    return model.replace("/", "-").replace(":", "-")


def _split_system(messages):
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [m for m in messages if m["role"] != "system"]
    return system, rest


class BaseLLM:
    def __init__(self, spec, temperature=None, max_tokens=4096):
        self.spec = spec
        self.provider, self.model = parse_spec(spec)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages, json_mode=False):
        raise NotImplementedError

    def generate(self, input_list, batch_size=None, **params):
        """Same contract as RAGForensics' OpenAIGenerator.generate: list of message lists -> list of str.
        Failures become "" exactly like the original."""
        out = []
        for messages in input_list:
            try:
                out.append(self.chat(messages))
            except Exception as e:  # noqa: BLE001 - mirror OpenAIGenerator's swallow-and-continue behaviour
                log.warning("%s generation failed: %s", self.spec, e)
                out.append("")
        return out


class ClaudeLLM(BaseLLM):
    def __init__(self, spec, effort=None, max_tokens=16000, **kw):
        super().__init__(spec, max_tokens=max_tokens, **kw)
        import anthropic

        self._anthropic = anthropic
        # Credentials: ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / `ant auth login` profile.
        self.client = anthropic.Anthropic(max_retries=5)
        self.effort = effort
        self.supports_effort = not self.model.startswith(_CLAUDE_NO_EFFORT_PREFIXES)
        self.use_fallbacks = self.model in _CLAUDE_FALLBACK_MODELS

    def chat(self, messages, json_mode=False):
        system, rest = _split_system(messages)
        if json_mode:
            system = (system + "\n\nRespond with a single JSON object only.").strip()
        params = dict(model=self.model, max_tokens=self.max_tokens, messages=rest)
        if system:
            params["system"] = system
        if self.effort and self.supports_effort:
            params["output_config"] = {"effort": self.effort}
        if self.use_fallbacks:
            params["betas"] = ["server-side-fallback-2026-07-01"]
            params["fallbacks"] = "default"
            response = self.client.beta.messages.create(**params)
        else:
            response = self.client.messages.create(**params)
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None) if response.stop_details else None
            log.warning("Claude refused (category=%s); returning empty output", category)
            return ""
        return "".join(b.text for b in response.content if b.type == "text")


class OllamaLLM(BaseLLM):
    def __init__(self, spec, num_ctx=8192, top_p=None, max_tokens=2048, **kw):
        super().__init__(spec, max_tokens=max_tokens, **kw)
        import ollama

        self.client = ollama.Client(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
        self.options = {"num_ctx": num_ctx, "num_predict": max_tokens}
        if self.temperature is not None:
            self.options["temperature"] = self.temperature
        if top_p is not None:
            self.options["top_p"] = top_p

    def chat(self, messages, json_mode=False):
        for attempt in range(3):
            try:
                r = self.client.chat(model=self.model, messages=messages,
                                     format="json" if json_mode else None, options=self.options)
                return r.message.content or ""
            except Exception as e:  # noqa: BLE001 - connection hiccups on a local server
                if attempt == 2:
                    raise
                log.warning("ollama error (%s), retrying", e)
                time.sleep(2 * (attempt + 1))


class OpenAILLM(BaseLLM):
    def __init__(self, spec, top_p=None, max_tokens=2048, **kw):
        super().__init__(spec, max_tokens=max_tokens, **kw)
        from openai import OpenAI

        self.client = OpenAI(base_url=os.environ.get("OPENAI_API_URL"))
        self.top_p = top_p

    def chat(self, messages, json_mode=False):
        params = dict(model=self.model, messages=messages, max_tokens=self.max_tokens)
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        r = self.client.chat.completions.create(**params)
        return r.choices[0].message.content or ""


def make_llm(spec, temperature=None, top_p=None, effort=None, num_ctx=8192):
    """temperature/top_p apply to Ollama/OpenAI only (current Claude models reject sampling params);
    effort applies to Claude only."""
    provider, _ = parse_spec(spec)
    if provider == "claude":
        return ClaudeLLM(spec, effort=effort)
    if provider == "ollama":
        return OllamaLLM(spec, temperature=temperature, top_p=top_p, num_ctx=num_ctx)
    return OpenAILLM(spec, temperature=temperature, top_p=top_p)
