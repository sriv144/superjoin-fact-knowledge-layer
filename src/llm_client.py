"""
Lightweight LLM client configured strictly for NVIDIA NIM (Nemotron).
Uses NVIDIA's hosted OpenAI-compatible endpoint for structured fact extraction
and qualitative semantic reasoning with deterministic temperature settings.
"""

import json
import os
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """NVIDIA NIM client for structured extraction and semantic reasoning."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        # Read API key strictly from parameter or NVIDIA_API_KEY environment variable
        if api_key is not None:
            nvidia_key = api_key.strip() if api_key else None
        else:
            nvidia_key = os.getenv("NVIDIA_API_KEY")
        nvidia_base_url = base_url or os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        default_model = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")

        try:
            request_timeout = float(os.getenv("NVIDIA_REQUEST_TIMEOUT_SECONDS", "45"))
        except ValueError:
            request_timeout = 45.0

        if nvidia_key:
            self.provider = "nvidia_nim"
            self.model_name = model_name or default_model
            self.base_url = nvidia_base_url
            self.api_key = nvidia_key
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=request_timeout,
                max_retries=1,
            )
        else:
            self.provider = "none"
            self.model_name = "none"
            self.base_url = None
            self.api_key = None
            self._client = None

    def is_available(self) -> bool:
        """Check if NVIDIA NIM is configured and client is initialized."""
        return self.provider == "nvidia_nim" and self._client is not None

    @staticmethod
    def _clean_json_string(text: str) -> str:
        """Strip markdown code fences and extraneous text from JSON response."""
        cleaned = text.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if match:
            cleaned = match.group(1).strip()
        return cleaned

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a structured JSON dictionary from NVIDIA Nemotron.
        Includes single-retry repair logic if JSON is malformed.
        """
        if not self.is_available():
            raise RuntimeError(
                "Live processing requires NVIDIA_API_KEY. "
                "Add NVIDIA_API_KEY to the project .env file and rerun the validation."
            )

        raw_response = self._call_llm(prompt, system_prompt=system_prompt, json_mode=True)
        try:
            cleaned = self._clean_json_string(raw_response)
            return json.loads(cleaned)
        except Exception as first_err:
            # Single-turn repair attempt
            repair_prompt = (
                f"The following output was expected to be valid JSON but had errors:\n{raw_response}\n\n"
                f"Error: {str(first_err)}\n\n"
                f"Please fix it and output ONLY valid JSON without explanation:"
            )
            repair_response = self._call_llm(repair_prompt, system_prompt=system_prompt, json_mode=True)
            cleaned_repair = self._clean_json_string(repair_response)
            return json.loads(cleaned_repair)

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a short text completion for qualitative semantic reasoning."""
        if not self.is_available():
            raise RuntimeError(
                "Live processing requires NVIDIA_API_KEY. "
                "Add NVIDIA_API_KEY to the project .env file and rerun the validation."
            )
        return self._call_llm(prompt, system_prompt=system_prompt, json_mode=False).strip()

    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None, json_mode: bool = False) -> str:
        """Internal dispatch using OpenAI SDK with deterministic Nemotron settings."""
        if not self._client:
            raise RuntimeError("NVIDIA client is not initialized.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Deterministic settings for fact extraction (temperature = 0.0, non-streaming)
        base_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0,
            "stream": False,
        }

        # Disable extended thinking for Nemotron models where supported
        if "nemotron" in self.model_name.lower():
            base_kwargs["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }

        # Attempt with response_format={"type": "json_object"} if requested
        if json_mode:
            try:
                kwargs_json = dict(base_kwargs)
                kwargs_json["response_format"] = {"type": "json_object"}
                resp = self._client.chat.completions.create(**kwargs_json)
                return resp.choices[0].message.content or ""
            except Exception:
                # Fallback if specific hosted model does not support response_format
                pass

        resp = self._client.chat.completions.create(**base_kwargs)
        return resp.choices[0].message.content or ""
