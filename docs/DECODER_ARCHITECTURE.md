# Text Decoder Architecture - Decoder Logic Implementation

## Overview

The `CustomTextDecoder` has been transformed from an **encoder-only** architecture to a proper **decoder architecture** suitable for autoregressive sequence generation tasks like Optical Music Recognition.

## Key Changes

### 1. TransformerEncoder → TransformerDecoder

**Old (Encoder Logic):**
- Only self-attention
- No causal masking
- Single layer norm and MLP

**New (Decoder Logic):**
- **Masked self-attention** with causal masking (tokens can only attend to previous tokens)
- **Cross-attention** to attend to vision encoder outputs
- Three layer norms (for self-attention, cross-attention, and MLP)
- Proper residual connections for all sublayers

```python
class TransformerDecoder(nn.Module):
    def forward(self, x, encoder_output=None, self_attn_mask=None, cross_attn_mask=None):
        # 1. Masked self-attention (causal)
        x = x + self.masked_self_attention(self.ln1(x), mask=self_attn_mask, is_causal=True)
        
        # 2. Cross-attention (if encoder output provided)
        if encoder_output is not None:
            x = x + self.cross_attention(self.ln2(x), encoder_output, mask=cross_attn_mask)
        
        # 3. Feed-forward network
        x = x + self.mlp(self.ln3(x))
        return x
```

### 2. Causal Masking Support

**MultiheadAttention** now supports causal masking via `is_causal` parameter:

```python
# Creates lower triangular mask for autoregressive generation
if is_causal:
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    # Combines with padding mask if present
```

This ensures tokens can only attend to previous positions during generation.

### 3. Cross-Attention Components

New components for attending from decoder to encoder:

- **CrossAttentionHead**: Single head cross-attention
- **MultiheadCrossAttention**: Multi-head cross-attention module

```python
class CrossAttentionHead:
    def forward(self, query, key_value, mask=None):
        Q = self.query(query)  # From decoder
        K = self.key(key_value)  # From encoder
        V = self.value(key_value)  # From encoder
        # ... attention computation
```

### 4. Dual Output Modes

**CustomTextDecoder** now supports two output modes:

#### Mode 1: Embeddings (Contrastive Learning)
```python
output = decoder(text, mask=mask, return_embeddings=True)
# Returns: [batch_size, emb_dim]
# Normalized embeddings from last token
```

#### Mode 2: Logits (Autoregressive Generation)
```python
output = decoder(text, mask=mask, return_embeddings=False)
# Returns: [batch_size, seq_len, vocab_size]
# Logits for next token prediction
```

### 5. Encoder Integration

Decoder can now receive encoder outputs for cross-attention:

```python
output = decoder(
    text, 
    encoder_output=vision_features,  # [batch_size, src_seq_len, d_model]
    mask=text_mask,
    return_embeddings=False
)
```

## Architecture Comparison

### Before (Encoder)
```
Input → Embedding → Positional Encoding
  ↓
TransformerEncoder Layer (repeated n_layers times):
  - LayerNorm
  - Self-Attention (bidirectional)
  - LayerNorm
  - MLP
  ↓
Extract last token → Project to embedding space
```

### After (Decoder)
```
Input → Embedding → Positional Encoding
  ↓
TransformerDecoder Layer (repeated n_layers times):
  - LayerNorm
  - Masked Self-Attention (causal, autoregressive)
  - LayerNorm
  - Cross-Attention (to encoder output)
  - LayerNorm
  - MLP
  ↓
Option A: Extract last token → Project to embedding space
Option B: Project all tokens → Vocabulary logits
```

## Usage Examples

### 1. Basic Text Decoder (No Vision Encoder)

```python
decoder = CustomTextDecoder(
    vocab_size=5000,
    d_model=256,
    max_seq_len=512,
    n_layers=6,
    n_heads=8,
    emb_dim=256
)

# Autoregressive generation
text = torch.randint(0, 5000, (batch_size, seq_len))
mask = torch.ones(batch_size, seq_len)
logits = decoder(text, mask=mask, return_embeddings=False)
# logits shape: [batch_size, seq_len, vocab_size]
```

### 2. With Vision Encoder (OMR Task)

```python
# Vision encoder processes image
vision_output = vision_encoder(image)  # [batch, 196, 768]

# Project vision features to decoder dimension
vision_features = mlp(vision_output)  # [batch, 196, 256]

# Decoder generates music notation autoregressively
logits = decoder(
    text=music_tokens,
    encoder_output=vision_features,
    mask=text_mask,
    return_embeddings=False
)
```

### 3. Contrastive Learning

```python
# Get normalized embeddings for contrastive loss
embeddings = decoder(
    text=music_tokens,
    mask=text_mask,
    return_embeddings=True
)
# embeddings shape: [batch_size, emb_dim]

# Compute contrastive loss
loss = contrastive_loss(image_embeddings, embeddings)
```

## Test Coverage

All decoder components are tested:

- ✅ **PositionalEncoding** (2 tests)
- ✅ **AttentionHead** (3 tests)
- ✅ **CrossAttentionHead** (2 tests)
- ✅ **MultiheadAttention** (3 tests, including causal)
- ✅ **MultiheadCrossAttention** (2 tests)
- ✅ **TransformerDecoder** (3 tests, including with encoder)
- ✅ **CustomTextDecoder** (5 tests, both output modes)

**Total: 20 tests passing**

## Benefits of Decoder Architecture

1. **Causal Masking**: Prevents information leakage during training
2. **Autoregressive Generation**: Can generate sequences token-by-token
3. **Cross-Attention**: Can condition on visual features from encoder
4. **Flexible Output**: Supports both contrastive learning and generation
5. **Standard Architecture**: Follows transformer decoder conventions

## Key Methods

### TransformerDecoder.forward()
```python
def forward(self, x, encoder_output=None, self_attn_mask=None, cross_attn_mask=None)
```
- `x`: Decoder input [B, tgt_len, d_model]
- `encoder_output`: Optional encoder output [B, src_len, d_model]
- `self_attn_mask`: Padding mask for decoder [B, tgt_len]
- `cross_attn_mask`: Padding mask for encoder [B, src_len]

### CustomTextDecoder.forward()
```python
def forward(self, text, encoder_output=None, mask=None, return_embeddings=False)
```
- `text`: Token IDs [B, seq_len]
- `encoder_output`: Optional vision features [B, src_len, d_model]
- `mask`: Padding mask [B, seq_len]
- `return_embeddings`: If True, returns [B, emb_dim], else [B, seq_len, vocab_size]

## Running Tests

```bash
# Test decoder components
pytest tests/test_text_decoder.py -v

# Test full OMR system
pytest -v
```

## Migration Notes

If you have existing code using the old encoder-based decoder:

**Old:**
```python
output = decoder(text, mask=mask)
# Always returned [batch_size, emb_dim]
```

**New:**
```python
# For embeddings (same as before)
output = decoder(text, mask=mask, return_embeddings=True)

# For logits (new capability)
output = decoder(text, mask=mask, return_embeddings=False)
```

The default behavior is `return_embeddings=False` (returns logits), so you need to explicitly set `return_embeddings=True` to get the old behavior.

