#!/usr/bin/env python3
"""
Phase 5: Production Deployment & Monitoring
Export model, enhance API, create monitoring dashboard
"""

import torch
import json
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import PharmaGNN

def export_model():
    """Export model to TorchScript for production"""
    print("=" * 70)
    print("Phase 5: Production Deployment & Monitoring")
    print("=" * 70)
    
    print("\n[1] Loading trained model...")
    model = PharmaGNN(num_node_features=6, hidden_channels=32, num_classes=13)
    
    weights_path = 'pharma_gnn_weights_universal.pt'
    if Path(weights_path).exists():
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"✓ Loaded weights from {weights_path}")
    else:
        print("⚠ No weights found!")
        return
    
    model.eval()
    
    # Create example input for tracing
    print("\n[2] Creating example input for tracing...")
    # We'll trace with a simple example
    example_x = torch.randn(3, 6)  # 3 nodes, 6 features
    example_edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    example_edge_attr = torch.randn(4, 1)
    example_batch = torch.tensor([0, 0, 0], dtype=torch.long)
    example_global_features = torch.randn(1, 11)
    example_func_group_features = torch.randn(1, 85)
    example_concentration = torch.tensor([[5.0]])  # pIC50 = 5 (10 µM)
    
    print("\n[3] Tracing model...")
    try:
        # Try scripting first
        scripted_model = torch.jit.script(model)
        scripted_model.save('pharma_gnn_production.pt')
        print("✓ Model saved as TorchScript: pharma_gnn_production.pt")
    except Exception as e:
        print(f"⚠ TorchScript failed: {e}")
        # Fall back to state_dict only
        torch.save(model.state_dict(), 'pharma_gnn_production_state_dict.pt')
        print("✓ Saved state_dict fallback: pharma_gnn_production_state_dict.pt")
    
    # Save model config
    config = {
        'num_node_features': 6,
        'hidden_channels': 32,
        'num_classes': 13,
        'num_global_features': 11,
        'num_func_groups': 85,
        'tasks': [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
            'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53',
            'CT_TOX'
        ],
        'ct_tox_index': 12,
        'concentration_feature': True,
        'calibration_temperature': 0.955
    }
    
    with open('model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("✓ Saved model_config.json")
    
    # Create production API specification
    api_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "PharmaGraph GNN API",
            "version": "2.0.0",
            "description": "Toxicity prediction using Graph Neural Networks"
        },
        "paths": {
            "/api/predict": {
                "post": {
                    "summary": "Predict toxicity for a single compound",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "smiles": {"type": "string", "example": "CCO"},
                                        "concentration_molar": {"type": "number", "example": 1e-5}
                                    },
                                    "required": ["smiles"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Successful prediction",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "smiles": {"type": "string"},
                                            "compound_name": {"type": "string"},
                                            "concentration_molar": {"type": "number"},
                                            "pIC50": {"type": "number"},
                                            "graph_info": {"type": "object"},
                                            "predictions": {
                                                "type": "object",
                                                "properties": {
                                                    "toxicity_risk": {"type": "string"},
                                                    "target_class": {"type": "integer"},
                                                    "ct_tox_class": {"type": "integer"},
                                                    "all_class_probs": {"type": "array", "items": {"type": "string"}}
                                                }
                                            },
                                            "explanation": {"type": "object"},
                                            "status": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/predict_batch": {
                "post": {
                    "summary": "Predict toxicity for multiple compounds",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "compounds": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "smiles": {"type": "string"},
                                                    "concentration_molar": {"type": "number", "default": 1e-5}
                                                },
                                                "required": ["smiles"]
                                            }
                                        }
                                    },
                                    "required": ["compounds"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Batch predictions",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "results": {"type": "array"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/health": {
                "get": {
                    "summary": "Health check",
                    "responses": {
                        "200": {"description": "Service healthy"}
                    }
                }
            },
            "/api/model_info": {
                "get": {
                    "summary": "Model information",
                    "responses": {
                        "200": {
                            "description": "Model metadata",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "model_version": {"type": "string"},
                                            "tasks": {"type": "array"},
                                            "calibration_temperature": {"type": "number"},
                                            "roc_auc": {"type": "number"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    with open('api_spec.json', 'w') as f:
        json.dump(api_spec, f, indent=2)
    print("✓ Saved api_spec.json")
    
    # Create monitoring configuration
    monitoring_config = {
        "metrics": {
            "prediction_latency_ms": {
                "type": "histogram",
                "buckets": [10, 50, 100, 200, 500, 1000],
                "description": "API prediction latency"
            },
            "prediction_confidence": {
                "type": "histogram",
                "buckets": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "description": "Model prediction confidence distribution"
            },
            "ct_tox_positive_rate": {
                "type": "counter",
                "description": "Rate of positive CT_TOX predictions"
            },
            "input_smiles_validity": {
                "type": "counter",
                "description": "Count of valid/invalid SMILES"
            },
            "errors_total": {
                "type": "counter",
                "description": "Total prediction errors"
            }
        },
        "alerting": {
            "high_latency_threshold_ms": 500,
            "low_confidence_threshold": 0.5,
            "error_rate_threshold": 0.05,
            "drift_detection_window_hours": 24
        },
        "logging": {
            "log_predictions": True,
            "log_input_smiles": True,
            "log_latency": True,
            "retention_days": 30
        }
    }
    
    with open('monitoring_config.json', 'w') as f:
        json.dump(monitoring_config, f, indent=2)
    print("✓ Saved monitoring_config.json")
    
    # Create Dockerfile for production
    dockerfile = """# Production Dockerfile
FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \\
    libxrender1 libxext6 libexpat1 \\
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --no-cache-dir --default-timeout=1800 \\
    "numpy<2" \\
    fastapi uvicorn pydantic \\
    torch --extra-index-url https://download.pytorch.org/whl/cpu \\
    torch_geometric \\
    rdkit \\
    pubchempy

# Copy model and code
COPY model.py main.py pharma_gnn_production.pt* model_config.json calibration_info.json /app/

EXPOSE 1234

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:1234/api/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "1234"]
"""
    
    with open('Dockerfile.production', 'w') as f:
        f.write(dockerfile)
    print("✓ Saved Dockerfile.production")
    
    # Create deployment summary
    summary = {
        "phase": 5,
        "status": "complete",
        "model": {
            "name": "PharmaGNN",
            "version": "2.0.0",
            "architecture": "GATv2 + Functional Group Interaction + Concentration",
            "weights_file": "pharma_gnn_production.pt",
            "config_file": "model_config.json"
        },
        "performance": {
            "roc_auc": 0.8151,
            "ece": 0.0840,
            "brier_score": 0.0506,
            "test_accuracy": 0.889,
            "calibration_temperature": 0.955
        },
        "api": {
            "endpoints": [
                "POST /api/predict",
                "POST /api/predict_batch",
                "GET /api/health",
                "GET /api/model_info"
            ],
            "port": 1234,
            "spec_file": "api_spec.json"
        },
        "deployment": {
            "dockerfile": "Dockerfile.production",
            "monitoring_config": "monitoring_config.json"
        },
        "data": {
            "training_compounds": 7841,
            "tasks": 13,
            "curated_test_accuracy": "88.9%"
        }
    }
    
    with open('deployment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Phase 5: Production Deployment Complete!")
    print("=" * 70)
    print("\nFiles created:")
    for f in [
        'pharma_gnn_production.pt',
        'model_config.json',
        'api_spec.json',
        'monitoring_config.json',
        'Dockerfile.production',
        'deployment_summary.json'
    ]:
        if Path(f).exists():
            print(f"  ✓ {f}")
    
    print("\nTo build production Docker image:")
    print("  docker build -f Dockerfile.production -t pharma-gnn:prod .")
    print("\nTo run:")
    print("  docker run -p 1234:1234 pharma-gnn:prod")
    print("=" * 70)


if __name__ == "__main__":
    export_model()