# gemini_llm.py
import os
from typing import List, Optional
from dotenv import load_dotenv, find_dotenv
from langchain.llms.base import LLM
import google.generativeai as genai

_ = load_dotenv(find_dotenv())
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def get_completion(prompt: str, model: str = "gemini-2.5-flash-lite") -> str:
    resp = genai.GenerativeModel(model).generate_content(prompt)
    return getattr(resp, "text", "") or ""

class Gemini_LLM(LLM):
    model_type: str = "gemini"

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt: str, history: List = [], stop: Optional[List[str]] = None) -> str:
        return get_completion(prompt)
    @property
    def _identifying_params(self):
        return {"model": self.model_type}
