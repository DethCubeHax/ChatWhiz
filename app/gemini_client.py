from typing import Iterator, Optional

import google.generativeai as genai

from .config import GEMINI_API_KEY, GEMINI_MODEL


class GeminiClient:
    def __init__(self) -> None:
        self._model: Optional[genai.GenerativeModel] = None

    def get_model(self) -> genai.GenerativeModel:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        if self._model is None:
            genai.configure(api_key=GEMINI_API_KEY)
            self._model = genai.GenerativeModel(GEMINI_MODEL)
        return self._model

    def generate_text(self, prompt: str) -> str:
        response = self.get_model().generate_content(prompt)
        return (response.text or "").strip()

    def stream_text(self, prompt: str) -> Iterator[str]:
        response = self.get_model().generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text


gemini_client = GeminiClient()
