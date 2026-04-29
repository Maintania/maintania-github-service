from openai import OpenAI
from google import genai
import tiktoken
import os


class LLMClient:
    def __init__(self, openai_api_key=os.getenv("OPENAI_API_KEY"), gemini_api_key=os.getenv("GEMINI_API_KEY")):
        # OpenAI
        self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None

        # Gemini (NEW SDK)
        self.gemini_client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None

    # -------------------------
    # GENERATE
    # -------------------------
    def generate(self, provider: str, model: str, prompt: str, config: dict = None) -> dict:
        if provider == "gemini":
            response = self.gemini_client.models.generate_content(
                model=model,
                contents=prompt,
                config=config or {}
            )
            return {
                "text": response.text,
                "usage": None
            }
        elif provider == "openai":
            response = self.openai_client.responses.create(
                model=model,
                input=prompt
            )
            return {"text": response.output_text}

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    # -------------------------
    # OpenAI
    # -------------------------
    def _openai_generate(self, model, prompt):
        if not self.openai_client:
            raise ValueError("OpenAI client not configured")

        response = self.openai_client.responses.create(
            model=model,
            input=prompt
        )

        return {
            "text": response.output_text,
            "usage": getattr(response, "usage", None)
        }

    # -------------------------
    # Gemini (NEW SDK)
    # -------------------------
    def _gemini_generate(self, model, prompt):
        if not self.gemini_client:
            raise ValueError("Gemini client not configured")

        response = self.gemini_client.models.generate_content(
            model=model,
            contents=prompt
        )

        return {
            "text": response.text,
            "usage": None  # Gemini doesn't always return usage here
        }

    # -------------------------
    # TOKEN COUNT
    # -------------------------
    def count_tokens(self, provider: str, model: str, prompt: str) -> dict:
        if provider == "gemini":
            return self._gemini_count_tokens(model, prompt)

        elif provider == "openai":
            return self._openai_count_tokens(model, prompt)

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    # -------------------------
    # Gemini (accurate)
    # -------------------------
    def _gemini_count_tokens(self, model, prompt):
        if not self.gemini_client:
            raise ValueError("Gemini client not configured")

        token_info = self.gemini_client.models.count_tokens(
            model=model,
            contents=prompt
        )

        return {
            "input_tokens": token_info.total_tokens,
            "output_tokens": None,
            "total_tokens": token_info.total_tokens
        }

    # -------------------------
    # OpenAI (estimated)
    # -------------------------
    def _openai_count_tokens(self, model, prompt):
        try:
            encoding = tiktoken.encoding_for_model(model)
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")

        tokens = len(encoding.encode(prompt))

        return {
            "input_tokens": tokens,
            "output_tokens": None,
            "total_tokens": tokens,
            "note": "estimated"
        }