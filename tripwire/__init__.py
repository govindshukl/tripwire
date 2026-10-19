"""Tripwire: on-prem typed decisions about adverse news, built on Qwen3.5-4B. Apache-2.0."""
from .model import Tripwire, load_questions

__all__ = ["Tripwire", "load_questions"]
