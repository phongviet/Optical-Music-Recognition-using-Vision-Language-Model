"""
Tests for Custom Text Decoder Model
"""
import sys
import os
import pytest
import torch
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'models'))

from models.Text_decoder_model import (
    CustomTextDecoder,
    TransformerDecoder,
    MultiheadAttention,
    MultiheadCrossAttention,
    AttentionHead,
    CrossAttentionHead,
    PositionalEncoding
)


@pytest.fixture
def model_config():
    """Load model configuration from yaml file"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'model_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['text_decoder']


@pytest.fixture
def small_config():
    """Load small model variant configuration"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'model_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['model_variants']['small']['text_decoder']


class TestPositionalEncoding:
    """Test PositionalEncoding module"""

    def test_positional_encoding_creation(self):
        """Test if PositionalEncoding can be instantiated"""
        d_model = 256
        max_seq_len = 512

        pe = PositionalEncoding(d_model, max_seq_len)
        assert pe is not None
        assert pe.d_model == d_model
        assert pe.max_seq_len == max_seq_len

    def test_positional_encoding_forward(self):
        """Test forward pass of PositionalEncoding"""
        d_model = 256
        max_seq_len = 512
        batch_size = 4
        seq_len = 128

        pe = PositionalEncoding(d_model, max_seq_len)
        x = torch.randn(batch_size, seq_len, d_model)
        output = pe(x)

        assert output.shape == (batch_size, seq_len, d_model)


class TestAttentionHead:
    """Test AttentionHead module"""

    def test_attention_head_creation(self):
        """Test if AttentionHead can be instantiated"""
        d_model = 256
        qkv_dim = 64

        head = AttentionHead(d_model, qkv_dim)
        assert head is not None
        assert head.qkv_dim == qkv_dim

    def test_attention_head_forward(self):
        """Test forward pass of AttentionHead"""
        d_model = 256
        qkv_dim = 64
        batch_size = 4
        seq_len = 128

        head = AttentionHead(d_model, qkv_dim)
        x = torch.randn(batch_size, seq_len, d_model)
        output = head(x)

        assert output.shape == (batch_size, seq_len, qkv_dim)

    def test_attention_head_with_mask(self):
        """Test forward pass with mask"""
        d_model = 256
        qkv_dim = 64
        batch_size = 4
        seq_len = 128

        head = AttentionHead(d_model, qkv_dim)
        x = torch.randn(batch_size, seq_len, d_model)
        mask = torch.ones(batch_size, seq_len)
        output = head(x, mask=mask)

        assert output.shape == (batch_size, seq_len, qkv_dim)


class TestMultiheadAttention:
    """Test MultiheadAttention module"""

    def test_multihead_attention_creation(self):
        """Test if MultiheadAttention can be instantiated"""
        d_model = 256
        n_heads = 8

        mha = MultiheadAttention(d_model, n_heads)
        assert mha is not None
        assert mha.qkv_dim == d_model // n_heads

    def test_multihead_attention_forward(self):
        """Test forward pass of MultiheadAttention"""
        d_model = 256
        n_heads = 8
        batch_size = 4
        seq_len = 128

        mha = MultiheadAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        output = mha(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_multihead_attention_causal(self):
        """Test forward pass with causal masking"""
        d_model = 256
        n_heads = 8
        batch_size = 4
        seq_len = 128

        mha = MultiheadAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        output = mha(x, is_causal=True)

        assert output.shape == (batch_size, seq_len, d_model)


class TestCrossAttention:
    """Test Cross-Attention modules"""

    def test_cross_attention_head_creation(self):
        """Test if CrossAttentionHead can be instantiated"""
        d_model = 256
        qkv_dim = 64

        head = CrossAttentionHead(d_model, qkv_dim)
        assert head is not None
        assert head.qkv_dim == qkv_dim

    def test_cross_attention_head_forward(self):
        """Test forward pass of CrossAttentionHead"""
        d_model = 256
        qkv_dim = 64
        batch_size = 4
        tgt_seq_len = 64
        src_seq_len = 196

        head = CrossAttentionHead(d_model, qkv_dim)
        query = torch.randn(batch_size, tgt_seq_len, d_model)
        key_value = torch.randn(batch_size, src_seq_len, d_model)
        output = head(query, key_value)

        assert output.shape == (batch_size, tgt_seq_len, qkv_dim)

    def test_multihead_cross_attention_creation(self):
        """Test if MultiheadCrossAttention can be instantiated"""
        d_model = 256
        n_heads = 8

        mha = MultiheadCrossAttention(d_model, n_heads)
        assert mha is not None
        assert mha.qkv_dim == d_model // n_heads

    def test_multihead_cross_attention_forward(self):
        """Test forward pass of MultiheadCrossAttention"""
        d_model = 256
        n_heads = 8
        batch_size = 4
        tgt_seq_len = 64
        src_seq_len = 196

        mha = MultiheadCrossAttention(d_model, n_heads)
        query = torch.randn(batch_size, tgt_seq_len, d_model)
        key_value = torch.randn(batch_size, src_seq_len, d_model)
        output = mha(query, key_value)

        assert output.shape == (batch_size, tgt_seq_len, d_model)


class TestTransformerDecoder:
    """Test TransformerDecoder module"""

    def test_transformer_decoder_creation(self):
        """Test if TransformerDecoder can be instantiated"""
        d_model = 256
        n_heads = 8

        decoder = TransformerDecoder(d_model, n_heads)
        assert decoder is not None
        assert decoder.d_model == d_model
        assert decoder.n_heads == n_heads

    def test_transformer_decoder_forward(self):
        """Test forward pass of TransformerDecoder"""
        d_model = 256
        n_heads = 8
        batch_size = 4
        seq_len = 128

        decoder = TransformerDecoder(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        output = decoder(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_transformer_decoder_with_encoder_output(self):
        """Test forward pass with encoder output (cross-attention)"""
        d_model = 256
        n_heads = 8
        batch_size = 4
        tgt_seq_len = 64
        src_seq_len = 196  # e.g., vision encoder output

        decoder = TransformerDecoder(d_model, n_heads)
        x = torch.randn(batch_size, tgt_seq_len, d_model)
        encoder_output = torch.randn(batch_size, src_seq_len, d_model)
        output = decoder(x, encoder_output=encoder_output)

        assert output.shape == (batch_size, tgt_seq_len, d_model)


class TestCustomTextDecoder:
    """Test CustomTextDecoder model"""

    def test_decoder_creation_with_config(self, model_config):
        """Test if CustomTextDecoder can be instantiated with config"""
        decoder = CustomTextDecoder(
            vocab_size=model_config['vocab_size'],
            d_model=model_config['d_model'],
            max_seq_len=model_config['max_seq_len'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            emb_dim=model_config['emb_dim']
        )

        assert decoder is not None
        assert decoder.max_seq_len == model_config['max_seq_len']

    def test_decoder_creation_small_variant(self, small_config):
        """Test if CustomTextDecoder can be instantiated with small variant config"""
        decoder = CustomTextDecoder(
            vocab_size=small_config['vocab_size'],
            d_model=small_config['d_model'],
            max_seq_len=small_config['max_seq_len'],
            n_layers=small_config['n_layers'],
            n_heads=small_config['n_heads'],
            emb_dim=small_config['emb_dim']
        )

        assert decoder is not None
        assert decoder.max_seq_len == small_config['max_seq_len']

    def test_decoder_forward_pass(self, model_config):
        """Test forward pass of CustomTextDecoder"""
        batch_size = 4
        seq_len = 64

        decoder = CustomTextDecoder(
            vocab_size=model_config['vocab_size'],
            d_model=model_config['d_model'],
            max_seq_len=model_config['max_seq_len'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            emb_dim=model_config['emb_dim']
        )

        # Create dummy input
        text = torch.randint(0, model_config['vocab_size'], (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len)

        # Test embeddings output
        output_emb = decoder(text, mask=mask, return_embeddings=True)
        assert output_emb.shape == (batch_size, model_config['emb_dim'])

        # Test logits output
        output_logits = decoder(text, mask=mask, return_embeddings=False)
        assert output_logits.shape == (batch_size, seq_len, model_config['vocab_size'])

    def test_decoder_parameter_count(self, model_config):
        """Test that decoder has trainable parameters"""
        decoder = CustomTextDecoder(
            vocab_size=model_config['vocab_size'],
            d_model=model_config['d_model'],
            max_seq_len=model_config['max_seq_len'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            emb_dim=model_config['emb_dim']
        )

        param_count = sum(p.numel() for p in decoder.parameters())
        trainable_param_count = sum(p.numel() for p in decoder.parameters() if p.requires_grad)

        assert param_count > 0
        assert trainable_param_count > 0

    def test_decoder_different_batch_sizes(self, small_config):
        """Test decoder with different batch sizes"""
        decoder = CustomTextDecoder(
            vocab_size=small_config['vocab_size'],
            d_model=small_config['d_model'],
            max_seq_len=small_config['max_seq_len'],
            n_layers=small_config['n_layers'],
            n_heads=small_config['n_heads'],
            emb_dim=small_config['emb_dim']
        )

        seq_len = 32

        for batch_size in [1, 2, 4, 8]:
            text = torch.randint(0, small_config['vocab_size'], (batch_size, seq_len))
            mask = torch.ones(batch_size, seq_len)

            # Test with embeddings output
            output = decoder(text, mask=mask, return_embeddings=True)
            assert output.shape == (batch_size, small_config['emb_dim'])

            # Test with logits output
            output_logits = decoder(text, mask=mask, return_embeddings=False)
            assert output_logits.shape == (batch_size, seq_len, small_config['vocab_size'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

