"""Provider-agnostic model factory. Switching providers is env-only."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from codeit.settings import Settings, get_settings


def get_model(settings: Settings | None = None) -> BaseChatModel:
    """Return a chat model for the configured provider.

    Uses ``init_chat_model`` — the same provider-agnostic primitive Deep
    Agents uses internally — so base_url and key validation live in *our*
    code, where we control the error messages.
    """
    s = settings or get_settings()
    if s.llm_provider == "ollama":
        return init_chat_model(
            model=s.llm_model,
            model_provider="ollama",
            base_url=s.ollama_base_url,
        )
    if s.llm_provider == "openai":
        if not s.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai. "
                "Set it in .env or `export OPENAI_API_KEY=...`."
            )
        kwargs: dict = {
            "model": s.llm_model,
            "model_provider": "openai",
            "api_key": s.openai_api_key,
        }
        if s.openai_base_url:
            kwargs["base_url"] = s.openai_base_url
        return init_chat_model(**kwargs)
    raise ValueError(f"Unknown LLM_PROVIDER={s.llm_provider!r}. Use 'ollama' or 'openai'.")
