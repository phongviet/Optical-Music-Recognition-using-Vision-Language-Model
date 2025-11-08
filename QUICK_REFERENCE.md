# Quick Reference Card

## Installation
```bash
pip install -r requirements.txt
```

## Run Demo
```bash
python scripts/demo_model_creation.py
```

## Run All Tests
```bash
pytest -v
```

## Run Specific Tests
```bash
# Text decoder only (offline)
pytest tests/test_text_decoder.py -v

# OMR model (requires internet)
pytest tests/test_omr_model.py -v
```

## Create Model from Config

```python
import yaml
from src.models.Text_decoder_model import CustomTextDecoder
from src.models.OMR_model import OMRModel

# Load config
with open('configs/model_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create decoder
decoder_cfg = config['text_decoder']
decoder = CustomTextDecoder(
    vocab_size=decoder_cfg['vocab_size'],
    d_model=decoder_cfg['d_model'],
    max_seq_len=decoder_cfg['max_seq_len'],
    n_layers=decoder_cfg['n_layers'],
    n_heads=decoder_cfg['n_heads'],
    emb_dim=decoder_cfg['emb_dim']
)

# Create OMR model
omr_cfg = config['omr_model']
model = OMRModel(
    vision_encoder_name=omr_cfg['vision_encoder_name'],
    text_decoder_name=omr_cfg['text_decoder_name'],
    mlp_layers=omr_cfg['mlp_layers'],
    decoder_config=decoder_cfg
)
```

## Key Configuration Parameters

### Text Decoder
- `vocab_size`: Vocabulary size
- `d_model`: Model dimension
- `max_seq_len`: Maximum sequence length
- `n_layers`: Number of transformer layers
- `n_heads`: Number of attention heads
- `emb_dim`: Output embedding dimension

### OMR Model
- `vision_encoder_name`: HuggingFace model name
- `text_decoder_name`: "custom" for CustomTextDecoder
- `mlp_layers`: List of hidden layer dimensions

## Model Variants

Edit `configs/model_config.yaml` and use:
- `model_variants.small` - Fast, lightweight
- `model_variants.medium` - Balanced (default)
- `model_variants.large` - High performance

