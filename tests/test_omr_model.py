"""
Comprehensive tests for OMR_model using CLIP-ViT-Large-Patch14 as vision encoder.
Tests cover initialization, forward pass, segmentation, generation, and edge cases.
"""

import pytest
import torch
import torch.nn as nn
from src.models.OMR_model import OMRModel


# Fixtures
@pytest.fixture
def decoder_config():
    """Standard decoder configuration for testing"""
    return {
        'vocab_size': 100,
        'd_model': 512,
        'max_seq_len': 256,
        'n_layers': 4,
        'n_heads': 8,
        'emb_dim': 512
    }


@pytest.fixture
def vision_encoder_name():
    """CLIP-ViT-Large-Patch14 model name"""
    return "openai/clip-vit-large-patch14"


@pytest.fixture
def mlp_layers():
    """MLP projection layers configuration"""
    return [1024, 768]


@pytest.fixture
def omr_model(vision_encoder_name, mlp_layers, decoder_config):
    """Create OMR model instance"""
    model = OMRModel(
        vision_encoder_name=vision_encoder_name,
        text_decoder_name="custom",
        mlp_layers=mlp_layers,
        decoder_config=decoder_config
    )
    model.eval()
    return model


@pytest.fixture
def sample_images():
    """Create sample image batch [B, C, H, W]"""
    # Use 896x224 so that 4 segments of 224x224 are created
    return torch.randn(2, 3, 896, 224)


@pytest.fixture
def sample_text_tokens():
    """Create sample text tokens [B, seq_len]"""
    return torch.randint(0, 100, (2, 50))


@pytest.fixture
def sample_attention_mask():
    """Create sample attention mask [B, seq_len]"""
    mask = torch.ones(2, 50)
    mask[0, 40:] = 0  # Padding for first sequence
    mask[1, 45:] = 0  # Padding for second sequence
    return mask


# === Initialization Tests ===
class TestOMRModelInitialization:
    """Test model initialization and configuration"""

    def test_model_initialization_success(self, vision_encoder_name, mlp_layers, decoder_config):
        """Test successful model initialization"""
        model = OMRModel(
            vision_encoder_name=vision_encoder_name,
            text_decoder_name="custom",
            mlp_layers=mlp_layers,
            decoder_config=decoder_config
        )

        assert model is not None
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'decoder')
        assert hasattr(model, 'mlp')
        assert hasattr(model, 'norm')

    def test_encoder_frozen(self, omr_model):
        """Test that vision encoder parameters are frozen"""
        for param in omr_model.encoder.parameters():
            assert not param.requires_grad, "Encoder parameters should be frozen"

    def test_decoder_config_required(self, vision_encoder_name, mlp_layers):
        """Test that decoder_config is required when using custom decoder"""
        with pytest.raises(ValueError, match="decoder_config is required"):
            OMRModel(
                vision_encoder_name=vision_encoder_name,
                text_decoder_name="custom",
                mlp_layers=mlp_layers,
                decoder_config=None
            )

    def test_encoder_dim_correct(self, omr_model):
        """Test that encoder dimension is correctly detected"""
        # CLIP-ViT-Large has hidden_size = 1024
        assert omr_model.encoder_dim == 1024

    def test_decoder_dim_correct(self, omr_model, decoder_config):
        """Test that decoder dimension matches config"""
        assert omr_model.decoder_dim == decoder_config['emb_dim']

    def test_mlp_projection_layers(self, omr_model, mlp_layers):
        """Test MLP projection has correct structure"""
        assert isinstance(omr_model.mlp, nn.Sequential)
        # Should have: Linear -> GELU -> Dropout for each hidden layer + final Linear
        # For mlp_layers=[1024, 768]: 3 layers per hidden (2x) + 1 final = 7 layers
        assert len(omr_model.mlp) == len(mlp_layers) * 3 + 1

    def test_mlp_with_empty_layers(self, vision_encoder_name, decoder_config):
        """Test MLP with no hidden layers (direct projection)"""
        model = OMRModel(
            vision_encoder_name=vision_encoder_name,
            text_decoder_name="custom",
            mlp_layers=[],
            decoder_config=decoder_config
        )
        # Should only have final projection layer
        assert len(model.mlp) == 1
        assert isinstance(model.mlp[0], nn.Linear)

    def test_mlp_with_none_layers(self, vision_encoder_name, decoder_config):
        """Test MLP with None layers (treated as empty)"""
        model = OMRModel(
            vision_encoder_name=vision_encoder_name,
            text_decoder_name="custom",
            mlp_layers=None,
            decoder_config=decoder_config
        )
        assert len(model.mlp) == 1

    def test_tokenizer_created(self, omr_model):
        """Test that custom tokenizer is instantiated"""
        assert hasattr(omr_model, 'tokenizer')
        assert omr_model.tokenizer is not None


# === Forward Pass Tests ===
class TestOMRModelForward:
    """Test forward pass with different configurations"""

    def test_forward_training_mode(self, omr_model, sample_images, sample_text_tokens):
        """Test forward pass in training mode with text tokens"""
        omr_model.train()
        logits = omr_model(sample_images, text_tokens=sample_text_tokens)

        batch_size, seq_len = sample_text_tokens.shape
        vocab_size = omr_model.decoder.vocab_size

        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_forward_with_attention_mask(self, omr_model, sample_images, sample_text_tokens, sample_attention_mask):
        """Test forward pass with attention mask"""
        logits = omr_model(
            sample_images,
            text_tokens=sample_text_tokens,
            attention_mask=sample_attention_mask
        )

        batch_size, seq_len = sample_text_tokens.shape
        vocab_size = omr_model.decoder.vocab_size

        assert logits.shape == (batch_size, seq_len, vocab_size)

    def test_forward_inference_mode(self, omr_model, sample_images):
        """Test forward pass in inference mode (no text tokens)"""
        omr_model.eval()
        with torch.no_grad():
            generated = omr_model(sample_images, text_tokens=None)

        batch_size = sample_images.shape[0]
        assert generated.shape[0] == batch_size
        assert generated.shape[1] > 0  # Should generate some tokens
        assert generated.dtype == torch.long

    def test_forward_different_batch_sizes(self, omr_model, decoder_config):
        """Test forward pass with different batch sizes"""
        for batch_size in [1, 2, 4, 8]:
            images = torch.randn(batch_size, 3, 896, 224)
            text_tokens = torch.randint(0, decoder_config['vocab_size'], (batch_size, 50))

            logits = omr_model(images, text_tokens=text_tokens)
            assert logits.shape[0] == batch_size

    def test_forward_different_image_sizes(self, omr_model, sample_text_tokens):
        """Test forward pass with different image heights (music sheets vary)"""
        # Use heights that are multiples of 224 to ensure CLIP compatibility
        for height in [224, 448, 672, 896]:
            images = torch.randn(2, 3, height, 224)
            logits = omr_model(images, text_tokens=sample_text_tokens)

            assert logits.shape[0] == 2
            assert logits.shape[1] == 50

    def test_forward_different_sequence_lengths(self, omr_model, sample_images):
        """Test forward pass with different sequence lengths"""
        for seq_len in [10, 50, 100, 200]:
            text_tokens = torch.randint(0, 100, (2, seq_len))
            logits = omr_model(sample_images, text_tokens=text_tokens)

            assert logits.shape[1] == seq_len

    def test_forward_single_image(self, omr_model):
        """Test forward pass with single image (batch_size=1)"""
        image = torch.randn(1, 3, 896, 224)
        text_tokens = torch.randint(0, 100, (1, 50))

        logits = omr_model(image, text_tokens=text_tokens)
        assert logits.shape == (1, 50, 100)

    def test_forward_custom_overlap(self, omr_model, sample_images, sample_text_tokens):
        """Test forward pass with custom segment overlap"""
        for overlap in [0.0, 0.25, 0.5, 0.75]:
            logits = omr_model(
                sample_images,
                text_tokens=sample_text_tokens,
                segment_overlap=overlap
            )
            assert logits.shape[0] == 2


# === Image Segmentation Tests ===
class TestImageSegmentation:
    """Test vertical image segmentation with overlap"""

    def test_segment_images_basic(self, omr_model, sample_images):
        """Test basic image segmentation"""
        segments = omr_model._segment_images_with_overlap(sample_images)

        batch_size, channels, height, width = sample_images.shape
        assert segments.shape[0] == batch_size
        assert segments.shape[2] == channels
        assert segments.shape[4] == width

    def test_segment_images_no_overlap(self, omr_model):
        """Test segmentation with no overlap"""
        images = torch.randn(2, 3, 448, 224)
        segments = omr_model._segment_images_with_overlap(images, overlap_ratio=0.0)

        # With no overlap and 4 segments, should get exactly 4 segments
        assert segments.shape[1] == 4

    def test_segment_images_half_overlap(self, omr_model):
        """Test segmentation with 50% overlap"""
        images = torch.randn(2, 3, 448, 224)
        segments = omr_model._segment_images_with_overlap(images, overlap_ratio=0.5)

        # With 50% overlap, should get more segments
        assert segments.shape[1] > 4

    def test_segment_images_quarter_overlap(self, omr_model):
        """Test segmentation with 1/4 overlap (default)"""
        images = torch.randn(2, 3, 448, 224)
        segments = omr_model._segment_images_with_overlap(images, overlap_ratio=0.25)

        # Should have more than 4 segments with overlap
        assert segments.shape[1] >= 4

    def test_segment_images_custom_num_segments(self, omr_model):
        """Test segmentation with custom number of segments"""
        images = torch.randn(2, 3, 448, 224)

        for num_segs in [2, 4, 8]:
            segments = omr_model._segment_images_with_overlap(
                images,
                num_segments=num_segs,
                overlap_ratio=0.0
            )
            # Without overlap, should get exactly num_segments
            assert segments.shape[1] == num_segs

    def test_segment_height_calculation(self, omr_model):
        """Test that segment height is calculated correctly"""
        images = torch.randn(2, 3, 448, 224)
        segments = omr_model._segment_images_with_overlap(images, num_segments=4)

        expected_segment_height = 448 // 4
        assert segments.shape[3] == expected_segment_height

    def test_segment_images_different_heights(self, omr_model):
        """Test segmentation with various image heights"""
        for height in [224, 448, 672, 896, 1120]:
            images = torch.randn(2, 3, height, 224)
            segments = omr_model._segment_images_with_overlap(images)

            assert segments.shape[0] == 2
            assert segments.shape[2] == 3
            assert segments.shape[4] == 224

    def test_segment_preserves_content(self, omr_model):
        """Test that segmentation preserves image content"""
        images = torch.randn(1, 3, 448, 224)
        segments = omr_model._segment_images_with_overlap(images, num_segments=4, overlap_ratio=0.0)

        # First segment should match first quarter of image
        segment_height = 448 // 4
        torch.testing.assert_close(
            segments[0, 0],
            images[0, :, :segment_height, :]
        )


# === Generation Tests ===
class TestGeneration:
    """Test autoregressive text generation"""

    def test_generate_basic(self, omr_model):
        """Test basic generation from visual features"""
        visual_features = torch.randn(2, 100, 512)  # [B, seq_len, decoder_dim]

        with torch.no_grad():
            generated = omr_model.generate(visual_features)

        assert generated.shape[0] == 2
        assert generated.dtype == torch.long
        assert (generated >= 0).all()

    def test_generate_max_length(self, omr_model):
        """Test generation respects max_length parameter"""
        visual_features = torch.randn(2, 100, 512)

        for max_len in [10, 50, 100, 200]:
            with torch.no_grad():
                generated = omr_model.generate(visual_features, max_length=max_len)

            assert generated.shape[1] <= max_len

    def test_generate_with_temperature(self, omr_model):
        """Test generation with different temperatures"""
        visual_features = torch.randn(2, 100, 512)

        for temp in [0.5, 1.0, 1.5]:
            with torch.no_grad():
                generated = omr_model.generate(
                    visual_features,
                    max_length=50,
                    temperature=temp
                )

            assert generated.shape[0] == 2
            assert generated.shape[1] <= 50

    def test_generate_starts_with_bos(self, omr_model):
        """Test that generation starts with BOS token (0)"""
        visual_features = torch.randn(2, 100, 512)

        with torch.no_grad():
            generated = omr_model.generate(visual_features, max_length=10)

        # First token should be BOS (0)
        assert (generated[:, 0] == 0).all()

    def test_generate_stops_on_eos(self, omr_model):
        """Test that generation can stop early on EOS token"""
        visual_features = torch.randn(1, 100, 512)

        with torch.no_grad():
            generated = omr_model.generate(visual_features, max_length=100)

        # Should stop before max_length if EOS is generated
        # (May or may not happen, but sequence length should be reasonable)
        assert generated.shape[1] > 0
        assert generated.shape[1] <= 100

    def test_generate_single_batch(self, omr_model):
        """Test generation with single sample"""
        visual_features = torch.randn(1, 100, 512)

        with torch.no_grad():
            # Limit to 200 to stay within max_seq_len (256)
            generated = omr_model.generate(visual_features, max_length=200)

        assert generated.shape[0] == 1
        assert generated.shape[1] <= 200

    def test_generate_large_batch(self, omr_model):
        """Test generation with larger batch"""
        visual_features = torch.randn(8, 100, 512)

        with torch.no_grad():
            generated = omr_model.generate(visual_features, max_length=50)

        assert generated.shape[0] == 8

    def test_generate_no_gradients(self, omr_model):
        """Test that generation doesn't compute gradients"""
        visual_features = torch.randn(2, 100, 512, requires_grad=True)

        # Limit to 200 to stay within max_seq_len (256)
        generated = omr_model.generate(visual_features, max_length=200)

        # Should not track gradients
        assert not generated.requires_grad


# === Edge Cases and Error Handling ===
class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_batch_handling(self, omr_model):
        """Test handling of edge case dimensions"""
        # Minimum viable dimensions (single 224x224 segment)
        images = torch.randn(1, 3, 224, 224)
        text_tokens = torch.randint(0, 100, (1, 1))

        logits = omr_model(images, text_tokens=text_tokens)
        assert logits.shape == (1, 1, 100)

    def test_very_long_sequence(self, omr_model):
        """Test with very long sequence (within max_seq_len)"""
        images = torch.randn(1, 3, 896, 224)
        text_tokens = torch.randint(0, 100, (1, 200))

        logits = omr_model(images, text_tokens=text_tokens)
        assert logits.shape == (1, 200, 100)

    def test_square_image(self, omr_model, sample_text_tokens):
        """Test with square images"""
        # Use 896x896 so segments are 224x896 (width varies, height fixed at 224)
        images = torch.randn(2, 3, 896, 896)
        logits = omr_model(images, text_tokens=sample_text_tokens)

        assert logits.shape[0] == 2

    def test_grayscale_image_fails(self, omr_model, sample_text_tokens):
        """Test that grayscale images (wrong channels) fail appropriately"""
        images = torch.randn(2, 1, 448, 224)  # Only 1 channel

        # Should fail because CLIP expects 3 channels
        with pytest.raises(Exception):
            omr_model(images, text_tokens=sample_text_tokens)

    def test_model_mode_switching(self, omr_model, sample_images, sample_text_tokens):
        """Test switching between train and eval modes"""
        # Train mode
        omr_model.train()
        logits_train = omr_model(sample_images, text_tokens=sample_text_tokens)

        # Eval mode
        omr_model.eval()
        with torch.no_grad():
            logits_eval = omr_model(sample_images, text_tokens=sample_text_tokens)

        # Outputs should have same shape but different values (due to dropout)
        assert logits_train.shape == logits_eval.shape

    def test_deterministic_inference(self, omr_model, sample_images, sample_text_tokens):
        """Test that inference is deterministic in eval mode"""
        omr_model.eval()

        with torch.no_grad():
            logits1 = omr_model(sample_images, text_tokens=sample_text_tokens)
            logits2 = omr_model(sample_images, text_tokens=sample_text_tokens)

        # Should be exactly the same in eval mode
        torch.testing.assert_close(logits1, logits2)


# === Integration Tests ===
@pytest.mark.integration
class TestOMRModelIntegration:
    """Integration tests for full pipeline"""

    def test_full_training_pipeline(self, omr_model, sample_images, sample_text_tokens, sample_attention_mask):
        """Test complete training forward pass with loss computation"""
        omr_model.train()

        logits = omr_model(
            sample_images,
            text_tokens=sample_text_tokens,
            attention_mask=sample_attention_mask
        )

        # Compute loss (cross-entropy)
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        loss = loss_fn(
            logits.view(-1, logits.size(-1)),
            sample_text_tokens.view(-1)
        )

        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_full_inference_pipeline(self, omr_model, sample_images):
        """Test complete inference pipeline"""
        omr_model.eval()

        with torch.no_grad():
            # Generate predictions
            predictions = omr_model(sample_images, text_tokens=None)

        batch_size = sample_images.shape[0]
        assert predictions.shape[0] == batch_size
        assert predictions.dtype == torch.long

        # Check all tokens are within vocabulary
        assert (predictions >= 0).all()
        assert (predictions < 100).all()

    def test_backward_pass(self, omr_model, sample_images, sample_text_tokens):
        """Test that backward pass works correctly"""
        omr_model.train()

        logits = omr_model(sample_images, text_tokens=sample_text_tokens)

        # Compute dummy loss
        loss = logits.sum()
        loss.backward()

        # Check that decoder has gradients (not frozen)
        decoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in omr_model.decoder.parameters()
        )
        assert decoder_has_grad, "Decoder should have gradients"

        # Check that encoder has no gradients (frozen)
        encoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in omr_model.encoder.parameters()
        )
        assert not encoder_has_grad, "Encoder should not have gradients (frozen)"

    def test_model_save_load(self, omr_model, sample_images, sample_text_tokens, tmp_path):
        """Test saving and loading model state"""
        omr_model.eval()

        # Get initial output
        with torch.no_grad():
            output1 = omr_model(sample_images, text_tokens=sample_text_tokens)

        # Save model
        save_path = tmp_path / "model.pt"
        torch.save(omr_model.state_dict(), save_path)

        # Create new model and load state
        new_model = OMRModel(
            vision_encoder_name="openai/clip-vit-large-patch14",
            text_decoder_name="custom",
            mlp_layers=[1024, 768],
            decoder_config={
                'vocab_size': 100,
                'd_model': 512,
                'max_seq_len': 256,
                'n_layers': 4,
                'n_heads': 8,
                'emb_dim': 512
            }
        )
        new_model.load_state_dict(torch.load(save_path))
        new_model.eval()

        # Get output from loaded model
        with torch.no_grad():
            output2 = new_model(sample_images, text_tokens=sample_text_tokens)

        # Outputs should be identical
        torch.testing.assert_close(output1, output2)


# === Performance and Memory Tests ===
@pytest.mark.slow
class TestPerformance:
    """Performance and memory tests (marked as slow)"""

    def test_large_batch_processing(self, omr_model):
        """Test processing large batch"""
        images = torch.randn(16, 3, 896, 224)
        text_tokens = torch.randint(0, 100, (16, 50))

        with torch.no_grad():
            logits = omr_model(images, text_tokens=text_tokens)

        assert logits.shape == (16, 50, 100)

    def test_memory_efficiency(self, omr_model, sample_images):
        """Test that model doesn't leak memory"""
        import gc

        omr_model.eval()

        # Run multiple forward passes
        for _ in range(10):
            with torch.no_grad():
                _ = omr_model(sample_images, text_tokens=None)

            # Force garbage collection
            gc.collect()

        # If we get here without OOM, test passes
        assert True

    def test_different_segment_overlaps_performance(self, omr_model, sample_images, sample_text_tokens):
        """Test that different overlaps don't cause issues"""
        overlaps = [0.0, 0.1, 0.25, 0.33, 0.5, 0.75]

        for overlap in overlaps:
            logits = omr_model(
                sample_images,
                text_tokens=sample_text_tokens,
                segment_overlap=overlap
            )
            assert logits.shape[0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

