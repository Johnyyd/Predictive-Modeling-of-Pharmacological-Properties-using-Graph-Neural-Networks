"""
PharmaGraph: AI Pharmacological Properties and Toxicity Prediction Engine.
"""

from pharma_gnn.model import PharmaGNN, FunctionalGroupInteraction
from pharma_gnn.chemistry import (
    smiles_to_graph,
    get_atom_features,
    get_compound_name,
    analyze_toxicophores,
    get_toxicophore_density,
    TOXICOPHORE_DEFINITIONS,
    frag_names,
    frag_funcs
)
from pharma_gnn.security import (
    SlidingWindowRateLimiter,
    RateLimitMiddleware,
    PayloadSizeLimitMiddleware,
    SecurityHeadersMiddleware,
    rate_limiter,
    DEFAULT_RATE_LIMIT,
    PREDICT_RATE_LIMIT
)
from pharma_gnn.config import load_model_config
from pharma_gnn.visualization import (
    plot_pretrain_progression,
    plot_finetune_curves,
    BENCHMARK_PRETRAIN_HISTORY,
    BENCHMARK_FINETUNE_HISTORY
)

__version__ = "2.1.0"
__all__ = [
    "PharmaGNN",
    "FunctionalGroupInteraction",
    "smiles_to_graph",
    "get_atom_features",
    "get_compound_name",
    "analyze_toxicophores",
    "get_toxicophore_density",
    "TOXICOPHORE_DEFINITIONS",
    "frag_names",
    "frag_funcs",
    "SlidingWindowRateLimiter",
    "RateLimitMiddleware",
    "PayloadSizeLimitMiddleware",
    "SecurityHeadersMiddleware",
    "rate_limiter",
    "DEFAULT_RATE_LIMIT",
    "PREDICT_RATE_LIMIT",
    "load_model_config",
    "plot_pretrain_progression",
    "plot_finetune_curves",
    "BENCHMARK_PRETRAIN_HISTORY",
    "BENCHMARK_FINETUNE_HISTORY",
]
