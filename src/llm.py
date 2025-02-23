"""
# Usage example:
    ```python
# Create and initialize a Gemini provider
llm = LLMFactory.create_provider('gemini')
llm.initialize(api_key='YOUR_API_KEY')

# Generate text
response = llm.generate(
    prompt="How does AI work?",
    temperature=0.7,
    max_tokens=100
)

print(response.text)
```
"""
from google import genai
from google.genai import Client
from abc import ABC, abstractmethod
from typing import Any, Dict
from dataclasses import dataclass

@dataclass
class LLMResponse:
    """Standardized response format for all LLM implementations"""
    text: str
    raw_response: Any  # Store the original response object
    metadata: Dict[str, Any] | None = None

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def initialize(self, api_key: str, **kwargs):
        """Initialize the LLM client with necessary credentials"""
        pass

    @abstractmethod
    def generate(self,
                prompt: str,
                model: str | None= None,
                temperature: float | None = None,
                max_tokens: int | None = None,
                **kwargs) -> LLMResponse:
        """Generate text from the LLM"""
        pass

class GeminiProvider(LLMProvider):
    """Concrete implementation for Google's Gemini"""

    def __init__(self):
        self.client: Client | None = None

    def initialize(self, api_key: str, **kwargs):
        """Initialize Gemini client with API key"""
        self.client = genai.Client(api_key=api_key)

    def generate(self,
                prompt: str,
                model: str | None = "gemini-pro",
                temperature: float | None = 0.7,
                max_tokens: int | None = None,
                **kwargs) -> LLMResponse:
        """Generate text using Gemini"""
        if not self.client:
            raise RuntimeError("Gemini client not initialized. Call initialize() first.")

        generation_config = {}
        if temperature is not None:
            generation_config['temperature'] = temperature
        if max_tokens is not None:
            generation_config['max_output_tokens'] = max_tokens

        response = self.client.models.generate_content(
            model=model if model else "gemini-pro",
            contents=[prompt],
            # generation_config=generation_config
        )

        return LLMResponse(
            text=response.text if response.text else "",
            raw_response=response,
            metadata={
                'model': model,
                'temperature': temperature,
                'max_tokens': max_tokens
            }
        )

class LLMFactory:
    """Factory class to create LLM providers"""

    @staticmethod
    def create_provider(provider_name: str) -> LLMProvider:
        providers = {
            'gemini': GeminiProvider
        }

        if provider_name not in providers:
            raise ValueError(f"Unknown provider: {provider_name}")

        return providers[provider_name]()
