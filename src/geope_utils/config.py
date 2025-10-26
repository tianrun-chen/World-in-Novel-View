"""
Configuration utilities for GeoPE components.

This module provides base configuration classes and utilities
for instantiating components with proper configuration management.
"""

from abc import ABC, abstractmethod
from typing import Any, Type, TypeVar

T = TypeVar('T')


class InstantiateConfig(ABC):
    """Base class for configuration objects that can instantiate components."""
    
    _target: Type
    
    @abstractmethod
    def setup(self) -> Any:
        """Setup and return the configured component."""
        pass
