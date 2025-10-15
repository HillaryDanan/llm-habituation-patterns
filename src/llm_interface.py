"""
LLM Interface - Unified API wrapper for Claude, GPT-4, and Gemini
Handles rate limiting, retries, and consistent response formatting
"""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import anthropic
import openai
import google.generativeai as genai

from config import (
    ANTHROPIC_API_KEY,
    OPENAI_API_KEY, 
    GOOGLE_API_KEY,
    MODELS,
    RATE_LIMITS
)

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized response format across all models"""
    model: str
    prompt: str
    response: str
    timestamp: datetime
    tokens_used: int
    metadata: Dict[str, Any]


class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request = 0.0
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        elapsed = time.time() - self.last_request
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        self.last_request = time.time()


class LLMInterface:
    """Unified interface for multiple LLM providers"""
    
    def __init__(self, model_name: str = "claude"):
        """
        Initialize LLM interface
        
        Args:
            model_name: One of "claude", "gpt4", "gemini"
        """
        if model_name not in MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model_name = model_name
        self.config = MODELS[model_name]
        self.provider = self.config["provider"]
        
        # Initialize client based on provider
        try:
            if self.provider == "anthropic":
                self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            elif self.provider == "openai":
                self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
            elif self.provider == "google":
                genai.configure(api_key=GOOGLE_API_KEY)
                self.client = genai.GenerativeModel(self.config["name"])
        except Exception as e:
            logger.error(f"Failed to initialize {self.provider} client: {e}")
            raise
        
        # Set up rate limiter
        rate_limit = RATE_LIMITS[self.provider]["requests_per_minute"]
        self.rate_limiter = RateLimiter(rate_limit)
        
        logger.info(f"Initialized {model_name} interface")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response from LLM
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (overrides default)
            max_tokens: Maximum tokens to generate (overrides default)
            **kwargs: Additional provider-specific parameters
        
        Returns:
            LLMResponse object
        """
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        # Use defaults if not specified
        temperature = temperature or self.config["temperature"]
        max_tokens = max_tokens or self.config["max_tokens"]
        
        # Call appropriate provider
        try:
            if self.provider == "anthropic":
                response = self._generate_anthropic(prompt, temperature, max_tokens, **kwargs)
            elif self.provider == "openai":
                response = self._generate_openai(prompt, temperature, max_tokens, **kwargs)
            elif self.provider == "google":
                response = self._generate_google(prompt, temperature, max_tokens, **kwargs)
            
            logger.debug(f"Generated response ({len(response.response)} chars)")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def _generate_anthropic(
        self, 
        prompt: str, 
        temperature: float, 
        max_tokens: int,
        **kwargs
    ) -> LLMResponse:
        """Generate using Anthropic Claude"""
        message = self.client.messages.create(
            model=self.config["name"],
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        
        return LLMResponse(
            model=self.model_name,
            prompt=prompt,
            response=message.content[0].text,
            timestamp=datetime.now(),
            tokens_used=message.usage.input_tokens + message.usage.output_tokens,
            metadata={
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
                "stop_reason": message.stop_reason,
            }
        )
    
    def _generate_openai(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> LLMResponse:
        """Generate using OpenAI GPT-4"""
        response = self.client.chat.completions.create(
            model=self.config["name"],
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return LLMResponse(
            model=self.model_name,
            prompt=prompt,
            response=response.choices[0].message.content,
            timestamp=datetime.now(),
            tokens_used=response.usage.total_tokens,
            metadata={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "finish_reason": response.choices[0].finish_reason,
            }
        )
    
    def _generate_google(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> LLMResponse:
        """Generate using Google Gemini"""
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            **kwargs
        )
        
        response = self.client.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        return LLMResponse(
            model=self.model_name,
            prompt=prompt,
            response=response.text,
            timestamp=datetime.now(),
            tokens_used=response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0,
            metadata={
                "prompt_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                "completion_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
                "finish_reason": response.candidates[0].finish_reason.name if response.candidates else None,
            }
        )
    
    def generate_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[LLMResponse]:
        """
        Generate responses for multiple prompts
        
        Args:
            prompts: List of input prompts
            **kwargs: Parameters passed to generate()
        
        Returns:
            List of LLMResponse objects
        """
        responses = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Generating response {i+1}/{len(prompts)}")
            try:
                response = self.generate(prompt, **kwargs)
                responses.append(response)
            except Exception as e:
                logger.error(f"Failed on prompt {i+1}: {e}")
                # Continue with remaining prompts
                continue
        
        return responses


def create_interface(model: str) -> LLMInterface:
    """Factory function to create LLM interface"""
    return LLMInterface(model)


if __name__ == "__main__":
    # Test the interface
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    test_prompt = "Explain the concept of habituation in 2-3 sentences."
    
    print("\n🧪 Testing LLM Interface\n")
    
    for model in ["claude", "gpt4", "gemini"]:
        try:
            print(f"Testing {model}...")
            interface = create_interface(model)
            response = interface.generate(test_prompt, max_tokens=200)
            
            print(f"✅ {model} response:")
            print(f"   {response.response[:100]}...")
            print(f"   Tokens: {response.tokens_used}\n")
            
        except Exception as e:
            print(f"❌ {model} failed: {e}\n")