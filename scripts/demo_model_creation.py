"""
Example script demonstrating how to create model instances from config
"""
import sys
import os
import yaml
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'models'))

from src.models.Text_decoder_model import CustomTextDecoder
from src.models.OMR_model import OMRModel


def load_config(config_path='configs/model_config.yaml'):
    """Load configuration from yaml file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_text_decoder(config):
    """Create CustomTextDecoder from config"""
    decoder_config = config['text_decoder']

    print("Creating CustomTextDecoder with configuration:")
    for key, value in decoder_config.items():
        print(f"  {key}: {value}")

    decoder = CustomTextDecoder(
        vocab_size=decoder_config['vocab_size'],
        d_model=decoder_config['d_model'],
        max_seq_len=decoder_config['max_seq_len'],
        n_layers=decoder_config['n_layers'],
        n_heads=decoder_config['n_heads'],
        emb_dim=decoder_config['emb_dim']
    )

    print(f"\n✓ CustomTextDecoder created successfully!")
    print(f"  Total parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in decoder.parameters() if p.requires_grad):,}")

    return decoder


def create_omr_model(config):
    """Create OMRModel from config"""
    omr_config = config['omr_model']
    decoder_config = config['text_decoder']

    print("\nCreating OMRModel with configuration:")
    print(f"  vision_encoder_name: {omr_config['vision_encoder_name']}")
    print(f"  text_decoder_name: {omr_config['text_decoder_name']}")
    print(f"  mlp_layers: {omr_config['mlp_layers']}")

    try:
        model = OMRModel(
            vision_encoder_name=omr_config['vision_encoder_name'],
            text_decoder_name=omr_config['text_decoder_name'],
            mlp_layers=omr_config['mlp_layers'],
            decoder_config=decoder_config
        )

        print(f"\n✓ OMRModel created successfully!")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"  Frozen parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")

        return model
    except Exception as e:
        print(f"\n✗ Failed to create OMRModel: {e}")
        print("  Note: This might require internet connection to download pretrained models")
        return None


def test_text_decoder_forward(decoder, config):
    """Test forward pass of CustomTextDecoder"""
    decoder_config = config['text_decoder']

    print("\n" + "="*60)
    print("Testing CustomTextDecoder forward pass...")
    print("="*60)

    batch_size = 4
    seq_len = 64

    # Create dummy input
    text = torch.randint(0, decoder_config['vocab_size'], (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len)

    print(f"Input shape: {text.shape}")
    print(f"Mask shape: {mask.shape}")

    # Test embeddings output (for contrastive learning)
    decoder.eval()
    with torch.no_grad():
        output_emb = decoder(text, mask=mask, return_embeddings=True)
    print(f"\nEmbeddings output shape: {output_emb.shape}")
    print(f"Expected shape: ({batch_size}, {decoder_config['emb_dim']})")
    assert output_emb.shape == (batch_size, decoder_config['emb_dim']), "Embeddings shape mismatch!"
    print("✓ Embeddings output successful!")

    # Test logits output (for autoregressive generation)
    with torch.no_grad():
        output_logits = decoder(text, mask=mask, return_embeddings=False)
    print(f"\nLogits output shape: {output_logits.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {decoder_config['vocab_size']})")
    assert output_logits.shape == (batch_size, seq_len, decoder_config['vocab_size']), "Logits shape mismatch!"
    print("✓ Logits output successful!")

    # Test with encoder output (cross-attention)
    encoder_seq_len = 196  # Typical vision encoder output
    encoder_output = torch.randn(batch_size, encoder_seq_len, decoder_config['d_model'])
    with torch.no_grad():
        output_with_encoder = decoder(text, encoder_output=encoder_output, mask=mask, return_embeddings=False)
    print(f"\nWith cross-attention shape: {output_with_encoder.shape}")
    assert output_with_encoder.shape == (batch_size, seq_len, decoder_config['vocab_size']), "Cross-attention shape mismatch!"
    print("✓ Cross-attention to encoder successful!")


def create_model_variants(config):
    """Create different model variants"""
    print("\n" + "="*60)
    print("Creating Model Variants...")
    print("="*60)

    variants = config.get('model_variants', {})

    for variant_name, variant_config in variants.items():
        print(f"\n{variant_name.upper()} variant:")
        decoder_cfg = variant_config['text_decoder']

        decoder = CustomTextDecoder(
            vocab_size=decoder_cfg['vocab_size'],
            d_model=decoder_cfg['d_model'],
            max_seq_len=decoder_cfg['max_seq_len'],
            n_layers=decoder_cfg['n_layers'],
            n_heads=decoder_cfg['n_heads'],
            emb_dim=decoder_cfg['emb_dim']
        )

        params = sum(p.numel() for p in decoder.parameters())
        print(f"  ✓ Created with {params:,} parameters")


def main():
    print("="*60)
    print("OMR Model Configuration and Testing Demo")
    print("="*60)

    # Load configuration
    config = load_config()
    print("\n✓ Configuration loaded successfully!")

    # Create CustomTextDecoder
    print("\n" + "="*60)
    print("1. Creating CustomTextDecoder")
    print("="*60)
    decoder = create_text_decoder(config)

    # Test CustomTextDecoder forward pass
    test_text_decoder_forward(decoder, config)

    # Create OMRModel
    print("\n" + "="*60)
    print("2. Creating OMRModel")
    print("="*60)
    omr_model = create_omr_model(config)

    # Create model variants
    create_model_variants(config)

    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)
    print("\nYou can now:")
    print("  1. Run tests: pytest tests/")
    print("  2. Use configs/model_config.yaml to configure your models")
    print("  3. Modify the config file for different model architectures")


if __name__ == "__main__":
    main()

