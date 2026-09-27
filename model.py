"""
PharmaGNN Root Entry Point.
Maintained for backward compatibility. Imports from modular pharma_gnn package.
"""
from pharma_gnn.model import PharmaGNN, FunctionalGroupInteraction

__all__ = ["PharmaGNN", "FunctionalGroupInteraction"]