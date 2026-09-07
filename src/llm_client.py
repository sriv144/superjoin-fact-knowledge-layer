"""
Unified, lightweight LLM client supporting Google Gemini and OpenAI-compatible providers.
Configured via environment variables; produces structured JSON outputs with automatic repair.
"""

import json
import os
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """Configurable LLM provider wrapper for structured extraction and semantic reasoning."""

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")
        
        if self.gemini_api_key:
            self.provider = "gemini"
            self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-flash-latest")
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_api_key)
            self._gemini_client = genai
        elif self.openai_api_key:
            self.provider = "openai"
            self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            from openai import OpenAI
            client_kwargs = {"api_key": self.openai_api_key}
            if self.openai_base_url:
                client_kwargs["base_url"] = self.openai_base_url
            self._openai_client = OpenAI(**client_kwargs)
        else:
            self.provider = "none"
            self.model_name = "none"

    def is_available(self) -> bool:
        """Check if a valid LLM provider is configured."""
        return self.provider in ("gemini", "openai")

    @staticmethod
    def _clean_json_string(text: str) -> str:
        """Strip markdown code fences and extraneous text from JSON response."""
        cleaned = text.strip()
        # Remove ```json ... ``` or ``` ... ```
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if match:
            cleaned = match.group(1).strip()
        return cleaned

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a structured JSON dictionary from the configured LLM.
        Includes single-retry repair logic if JSON is malformed.
        """
        if not self.is_available():
            raise RuntimeError("No LLM API key configured. Set GEMINI_API_KEY or OPENAI_API_KEY in .env")

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
        """Generate a short text completion for semantic reasoning."""
        if not self.is_available():
            raise RuntimeError("No LLM API key configured. Set GEMINI_API_KEY or OPENAI_API_KEY in .env")
        return self._call_llm(prompt, system_prompt=system_prompt, json_mode=False).strip()

    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None, json_mode: bool = False) -> str:
        """Internal dispatch to either Gemini or OpenAI."""
        if self.provider == "gemini":
            gen_config = {}
            if json_mode:
                gen_config["response_mime_type"] = "application/json"
            
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"System Instructions: {system_prompt}\n\nUser Request: {prompt}"
            
            model = self._gemini_client.GenerativeModel(
                model_name=self.model_name,
                generation_config=gen_config if gen_config else None
            )
            resp = model.generate_content(full_prompt)
            return resp.text or ""

        elif self.provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.1,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            resp = self._openai_client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""

        raise RuntimeError(f"Unknown provider: {self.provider}")
