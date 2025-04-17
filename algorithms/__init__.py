"""
Algorithms package for MuralBot.

This package contains advanced algorithms for dithering, fill pattern generation, and path optimization.
"""

from .dithering import apply_dithering
from .fill_patterns import generate_fill_pattern
from .path_optimization import optimize_path_sequence

__all__ = ["apply_dithering", "generate_fill_pattern", "optimize_path_sequence"]
