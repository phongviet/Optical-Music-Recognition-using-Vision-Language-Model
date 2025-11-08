from torch import nn
import torch
import numpy as np

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        # Layer norm for masked self-attention
        self.ln1 = nn.LayerNorm(d_model)
        # Masked self-attention (causal)
        self.masked_self_attention = MultiheadAttention(d_model, n_heads)

        # Layer norm for cross-attention
        self.ln2 = nn.LayerNorm(d_model)
        # Cross-attention to encoder outputs
        self.cross_attention = MultiheadCrossAttention(d_model, n_heads)

        # Layer norm for MLP
        self.ln3 = nn.LayerNorm(d_model)
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model)
        )

    def forward(self, x, encoder_output=None, self_attn_mask=None, cross_attn_mask=None):
        """
        Args:
            x: Decoder input [Batch_size, tgt_seq_len, d_model]
            encoder_output: Encoder output [Batch_size, src_seq_len, d_model] (optional)
            self_attn_mask: Causal mask for self-attention [Batch_size, tgt_seq_len]
            cross_attn_mask: Mask for cross-attention [Batch_size, src_seq_len]
        """
        # Masked self-attention with residual connection
        x_norm = self.ln1(x)
        self_attn_out = self.masked_self_attention(x_norm, mask=self_attn_mask, is_causal=True)
        x = x + self_attn_out

        # Cross-attention with residual connection (if encoder output is provided)
        if encoder_output is not None:
            x_norm = self.ln2(x)
            cross_attn_out = self.cross_attention(x_norm, encoder_output, mask=cross_attn_mask)
            x = x + cross_attn_out

        # Feed-forward network with residual connection
        x_norm = self.ln3(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out

        return x  # Output size: [Batch_size, tgt_seq_len, d_model]

class AttentionHead(nn.Module):
    def __init__(self, d_model, qkv_dim):
        super().__init__()

        self.qkv_dim = qkv_dim

        self.query = nn.Linear(d_model, qkv_dim)
        self.key = nn.Linear(d_model, qkv_dim)
        self.value = nn.Linear(d_model, qkv_dim)

    def forward(self, x, mask=None):
        # X.shape: [Batch_size, seq_len, d_model]
        Q = self.query(x)  # [Batch_size, seq_len, qkv_dim]
        K = self.key(x)
        V = self.value(x)

        attention = Q @ K.transpose(-2, -1)  # [Batch_size, seq_len, seq_len]
        attention /= (self.qkv_dim ** 0.5)

        if mask is not None:
            # Handle both 2D [B, seq_len] and 3D [B, seq_len, seq_len] masks
            if mask.dim() == 2:
                # Padding mask: [B, seq_len] -> [B, 1, seq_len]
                mask = mask.unsqueeze(1)
            # Apply mask
            attention = attention.masked_fill(mask == 0, float('-inf'))

        attention = attention.softmax(dim=-1)  # [Batch_size, seq_len, seq_len]

        attention = attention @ V  # [Batch_size, seq_len, qkv_dim]

        return attention


class CrossAttentionHead(nn.Module):
    """Cross-attention head for attending from decoder to encoder"""
    def __init__(self, d_model, qkv_dim):
        super().__init__()

        self.qkv_dim = qkv_dim

        self.query = nn.Linear(d_model, qkv_dim)
        self.key = nn.Linear(d_model, qkv_dim)
        self.value = nn.Linear(d_model, qkv_dim)

    def forward(self, query, key_value, mask=None):
        """
        Args:
            query: Decoder features [Batch_size, tgt_seq_len, d_model]
            key_value: Encoder features [Batch_size, src_seq_len, d_model]
            mask: Encoder padding mask [Batch_size, src_seq_len]
        """
        Q = self.query(query)  # [Batch_size, tgt_seq_len, qkv_dim]
        K = self.key(key_value)  # [Batch_size, src_seq_len, qkv_dim]
        V = self.value(key_value)  # [Batch_size, src_seq_len, qkv_dim]

        attention = Q @ K.transpose(-2, -1)  # [Batch_size, tgt_seq_len, src_seq_len]
        attention /= (self.qkv_dim ** 0.5)

        if mask is not None:
            # Encoder padding mask: [B, src_seq_len] -> [B, 1, src_seq_len]
            attention = attention.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        attention = attention.softmax(dim=-1)  # [Batch_size, tgt_seq_len, src_seq_len]

        attention = attention @ V  # [Batch_size, tgt_seq_len, qkv_dim]

        return attention

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.qkv_dim = d_model // n_heads

        self.W_o = nn.Linear(d_model, d_model)

        self.multiple_heads = nn.ModuleList([
            AttentionHead(d_model, self.qkv_dim) for _ in range(n_heads)
        ])

    def forward(self, x, mask=None, is_causal=False):
        """
        Args:
            x: Input tensor [Batch_size, seq_len, d_model]
            mask: Padding mask [Batch_size, seq_len]
            is_causal: If True, apply causal masking for autoregressive decoding
        """
        # Create causal mask if needed
        if is_causal:
            seq_len = x.size(1)
            # Create lower triangular matrix: [seq_len, seq_len]
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            if mask is not None:
                # Combine causal mask with padding mask
                # mask: [B, seq_len] -> [B, 1, seq_len] -> [B, seq_len, seq_len]
                padding_mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
                combined_mask = causal_mask.unsqueeze(0) * padding_mask
                mask = combined_mask
            else:
                mask = causal_mask.unsqueeze(0).expand(x.size(0), -1, -1)

        out = torch.cat(
            [head(x, mask=mask) for head in self.multiple_heads], dim=-1
        )  # [Batch_size, seq_len, d_model]

        out = self.W_o(out)  # [Batch_size, seq_len, d_model]

        return out


class MultiheadCrossAttention(nn.Module):
    """Cross-attention module for attending to encoder outputs"""
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.qkv_dim = d_model // n_heads

        self.W_o = nn.Linear(d_model, d_model)

        self.multiple_heads = nn.ModuleList([
            CrossAttentionHead(d_model, self.qkv_dim) for _ in range(n_heads)
        ])

    def forward(self, query, key_value, mask=None):
        """
        Args:
            query: Decoder features [Batch_size, tgt_seq_len, d_model]
            key_value: Encoder features [Batch_size, src_seq_len, d_model]
            mask: Encoder padding mask [Batch_size, src_seq_len]
        """
        out = torch.cat(
            [head(query, key_value, mask=mask) for head in self.multiple_heads], dim=-1
        )  # [Batch_size, tgt_seq_len, d_model]

        out = self.W_o(out)  # [Batch_size, tgt_seq_len, d_model]

        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1) # Shape [max_seq_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        ) # Shape [d_model // 2]

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class CustomTextDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, n_layers, n_heads, emb_dim):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.hidden_size = emb_dim  # Output embedding dimension
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = PositionalEncoding(d_model, max_seq_len)

        # Use TransformerDecoder layers instead of encoder
        self.transformer_decoder = nn.ModuleList([
            TransformerDecoder(d_model, n_heads) for _ in range(n_layers)
        ])

        # Output projection to vocabulary (for next token prediction)
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Embedding projection for contrastive learning (optional)
        self.embedding_projection = nn.Parameter(torch.randn(d_model, emb_dim))

    def forward(self, text, encoder_output=None, mask=None):
        """
        Args:
            text: Input token IDs [batch_size, seq_len]
            encoder_output: Vision encoder output [batch_size, src_seq_len, encoder_dim]
            mask: Padding mask for text [batch_size, seq_len]
            return_embeddings: If True, return normalized embeddings instead of logits
        Returns:
            logits [batch_size, seq_len, vocab_size]
        """
        # Ensure text and mask are [batch_size, seq_len]
        if text.dim() == 3 and text.shape[1] == 1:
            text = text.squeeze(1)
        if mask is not None and mask.dim() == 3 and mask.shape[1] == 1:
            mask = mask.squeeze(1)

        x = self.embed(text)  # [B, seq_len, d_model]
        x = self.positional_embedding(x)

        # Pass through decoder layers with causal masking
        for decoder_layer in self.transformer_decoder:
            x = decoder_layer(
                x,
                encoder_output=encoder_output,
                self_attn_mask=mask,
                cross_attn_mask=None  # Can add encoder mask if needed
            )

        # Return logits for next token prediction (autoregressive)
        logits = self.output_projection(x)  # [B, seq_len, vocab_size]
        return logits

class CustomTokenizer():
    pass