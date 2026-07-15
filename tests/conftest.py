"""
Shared fixtures for tests.
"""

import os

import pytest


def pytest_configure(config):
    """Check for required API keys before running tests."""
    # These are optional - tests will skip if not available
    pass


@pytest.fixture
def has_openai_key():
    """Check if OpenAI API key is available."""
    return os.getenv("OPENAI_API_KEY") is not None


@pytest.fixture
def has_openrouter_key():
    """Check if OpenRouter API key is available."""
    return os.getenv("OPENROUTER_API_KEY") is not None


@pytest.fixture
def simple_message():
    """A simple test message."""
    return [[{"role": "user", "content": "What is 2+2? Reply with just the number."}]]


@pytest.fixture
def simple_message_with_system():
    """A test message with system prompt."""
    return [
        [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "What is 2+2? Reply with just the number."},
        ]
    ]
