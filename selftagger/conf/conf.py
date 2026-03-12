
import os

class Config:
    def __init__(self):
        self.searxUrl = os.getenv('SELFTAGGER_SEARX_URL')
        if not self.searxUrl:
            raise ValueError("Environment variable SELFTAGGER_SEARX_URL is not set")

        self.llmBaseUrl = os.getenv('SELFTAGGER_LLM_BASE_URL')
        if not self.llmBaseUrl:
            raise ValueError("Environment variable SELFTAGGER_LLM_BASE_URL is not set")

        self.llmApiKey  = os.getenv('SELFTAGGER_LLM_API_KEY')
        if not self.llmApiKey:
            raise ValueError("Environment variable SELFTAGGER_LLM_API_KEY is not set")

        self.llmModel  = os.getenv('SELFTAGGER_LLM_MODEL')
        if not self.llmModel:
            raise ValueError("Environment variable SELFTAGGER_LLM_MODEL is not set")

        self.llmProject  = os.getenv('SELFTAGGER_LLM_PROJECT')
