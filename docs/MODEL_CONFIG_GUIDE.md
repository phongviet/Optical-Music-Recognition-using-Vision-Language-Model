# Model Configuration Guide

This guide explains how to use the model configuration system and run tests for the Optical Music Recognition project.

## Configuration Files

### Main Configuration: `configs/model_config.yaml`

This file contains all the configuration parameters for your models:

#### OMR Model Configuration
```yaml
omr_model:
  vision_encoder_name: "google/vit-base-patch16-224"  # HuggingFace model
  text_decoder_name: "custom"  # Use "custom" for CustomTextDecoder
  mlp_layers: [1024, 512, 256]  # Hidden dimensions for MLP projection
```

#### Text Decoder Configuration
```yaml
text_decoder:
  vocab_size: 5000      # Size of vocabulary
  d_model: 256          # Model dimension
  max_seq_len: 512      # Maximum sequence length
  n_layers: 6           # Number of transformer layers
  n_heads: 8            # Number of attention heads
  emb_dim: 256          # Output embedding dimension
```

#### Model Variants

The config also includes three pre-configured variants:
- **small**: Lightweight model for faster training/inference
- **medium**: Balanced model (default configuration)
- **large**: High-capacity model for best performance

## Creating Model Instances

### Option 1: Using Python Script

Run the demo script to see how to create models from config:

```bash
python scripts/demo_model_creation.py
```

### Option 2: In Your Code

```python
import yaml
import sys
sys.path.insert(0, 'src')

from src.models.Text_decoder_model import CustomTextDecoder
from src.models.OMR_model import OMRModel

# Load configuration
with open('configs/model_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create Text Decoder
decoder_config = config['text_decoder']
decoder = CustomTextDecoder(
    vocab_size=decoder_config['vocab_size'],
    d_model=decoder_config['d_model'],
    max_seq_len=decoder_config['max_seq_len'],
    n_layers=decoder_config['n_layers'],
    n_heads=decoder_config['n_heads'],
    emb_dim=decoder_config['emb_dim']
)

# Create OMR Model
omr_config = config['omr_model']
model = OMRModel(
    vision_encoder_name=omr_config['vision_encoder_name'],
    text_decoder_name=omr_config['text_decoder_name'],
    mlp_layers=omr_config['mlp_layers'],
    decoder_config=decoder_config
)
```

## Running Tests

### Install Dependencies

First, install all required packages:

```bash
pip install -r requirements.txt
```

### Run All Tests

```bash
pytest
```

or with more verbose output:

```bash
pytest -v
```

### Run Specific Test Files

Test the Text Decoder only (no internet required):
```bash
pytest tests/test_text_decoder.py -v
```

Test the OMR Model (requires internet for HuggingFace models):
```bash
pytest tests/test_omr_model.py -v
```

### Run Specific Test Classes

```bash
pytest tests/test_text_decoder.py::TestCustomTextDecoder -v
pytest tests/test_omr_model.py::TestOMRModel -v
```

### Run Specific Test Functions

```bash
pytest tests/test_text_decoder.py::TestCustomTextDecoder::test_decoder_creation_with_config -v
```

### Run with Coverage Report

```bash
pytest --cov=src --cov-report=html
```

This creates an HTML coverage report in `htmlcov/index.html`.

## Test Structure

### Text Decoder Tests (`test_text_decoder.py`)

Tests for individual components:
- **TestPositionalEncoding**: Tests positional encoding module
- **TestAttentionHead**: Tests single attention head
- **TestMultiheadAttention**: Tests multi-head attention
- **TestTransformerEncoder**: Tests transformer encoder layer
- **TestCustomTextDecoder**: Tests complete text decoder model

### OMR Model Tests (`test_omr_model.py`)

Tests for the complete OMR system:
- **TestOMRModel**: Tests model creation, architecture, and parameters
- **TestOMRModelIntegration**: Integration tests for the full model

## Key Features

### 1. Configuration-Driven Design
All model parameters are specified in YAML files, making it easy to:
- Switch between model variants
- Experiment with different architectures
- Share configurations with team members

### 2. Comprehensive Testing
Tests verify:
- ✓ Model instantiation from config
- ✓ Parameter counts (total and trainable)
- ✓ Forward pass functionality
- ✓ Correct tensor shapes
- ✓ Frozen encoder parameters
- ✓ Different batch sizes

### 3. Automatic Test Skipping
Tests that require downloading HuggingFace models will automatically skip if:
- No internet connection is available
- Models cannot be downloaded
- Dependencies are missing

## Modifying Configurations

### To Change Model Size

Edit `configs/model_config.yaml` and modify the relevant sections:

```yaml
text_decoder:
  d_model: 512  # Increase for larger model
  n_layers: 12  # More layers for deeper model
  n_heads: 16   # More attention heads
```

### To Use a Different Vision Encoder

```yaml
omr_model:
  vision_encoder_name: "microsoft/swin-base-patch4-window7-224"
```

### To Change MLP Architecture

```yaml
omr_model:
  mlp_layers: [2048, 1024, 512, 256]  # Add more layers
```

## Troubleshooting

### Import Errors
If you get import errors, make sure you're running from the project root:
```bash
cd "D:\OneDrive - Hanoi University of Science and Technology\Project\Optical Music Recognition"
```

### HuggingFace Model Download Issues
- Check internet connection
- Tests will skip automatically if models can't be downloaded
- You can still run text decoder tests offline

### Memory Issues
If you run out of memory:
- Use the "small" model variant
- Reduce batch size in tests
- Close other applications

## Next Steps

1. **Run the demo**: `python scripts/demo_model_creation.py`
2. **Run tests**: `pytest -v`
3. **Customize config**: Edit `configs/model_config.yaml`
4. **Add your own tests**: Create new test files in `tests/`

## File Structure

```
├── configs/
│   ├── config.yaml           # General project config
│   └── model_config.yaml     # Model-specific config (NEW)
├── tests/
│   ├── __init__.py           # Test package init (NEW)
│   ├── README.md             # Test documentation (NEW)
│   ├── test_text_decoder.py  # Text decoder tests (NEW)
│   └── test_omr_model.py     # OMR model tests (NEW)
├── scripts/
│   └── demo_model_creation.py  # Demo script (NEW)
├── src/
│   └── models/
│       ├── Text_decoder_model.py  # Updated with hidden_size
│       └── OMR_model.py           # Updated with decoder_config
├── pytest.ini                # Pytest configuration (NEW)
└── requirements.txt          # Updated with dependencies
```

## Contact & Support

For issues or questions:
1. Check this guide
2. Review test files for usage examples
3. Run demo script to see working examples

