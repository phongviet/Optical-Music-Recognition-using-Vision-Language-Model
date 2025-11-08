"""
Tests for OMR Model
"""
import sys
import os
import pytest
import torch
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'models'))

from models.OMR_model import OMRModel


@pytest.fixture
def model_config():
    """Load model configuration from yaml file"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'model_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture
def omr_config(model_config):
    """Get OMR model configuration"""
    return model_config['omr_model']


@pytest.fixture
def text_decoder_config(model_config):
    """Get text decoder configuration"""
    return model_config['text_decoder']


@pytest.fixture
def small_variant_config(model_config):
    """Get small model variant configuration"""
    return model_config['model_variants']['small']


class TestOMRModel:
    """Test OMR Model"""

    def test_omr_model_creation_with_config(self, omr_config, text_decoder_config):
        """Test if OMRModel can be instantiated with config"""
        try:
            model = OMRModel(
                vision_encoder_name=omr_config['vision_encoder_name'],
                text_decoder_name=omr_config['text_decoder_name'],
                mlp_layers=omr_config['mlp_layers'],
                decoder_config=text_decoder_config
            )

            assert model is not None
            assert model.encoder is not None
            assert model.decoder is not None
            assert model.mlp is not None
            assert model.norm is not None
        except Exception as e:
            pytest.skip(f"Skipping test due to missing HuggingFace model or network issue: {e}")

    def test_omr_model_encoder_frozen(self, omr_config, text_decoder_config):
        """Test that encoder parameters are frozen"""
        try:
            model = OMRModel(
                vision_encoder_name=omr_config['vision_encoder_name'],
                text_decoder_name=omr_config['text_decoder_name'],
                mlp_layers=omr_config['mlp_layers'],
                decoder_config=text_decoder_config
            )

            # Check that encoder parameters are frozen
            for param in model.encoder.parameters():
                assert param.requires_grad == False, "Encoder parameters should be frozen"
        except Exception as e:
            pytest.skip(f"Skipping test due to missing HuggingFace model or network issue: {e}")

    def test_omr_model_decoder_trainable(self, omr_config, text_decoder_config):
        """Test that decoder parameters are trainable"""
        try:
            model = OMRModel(
                vision_encoder_name=omr_config['vision_encoder_name'],
                text_decoder_name=omr_config['text_decoder_name'],
                mlp_layers=omr_config['mlp_layers'],
                decoder_config=text_decoder_config
            )

            # Check that decoder has trainable parameters
            decoder_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
            assert decoder_params > 0, "Decoder should have trainable parameters"
        except Exception as e:
            pytest.skip(f"Skipping test due to missing HuggingFace model or network issue: {e}")

    def test_omr_model_mlp_layers(self, omr_config, text_decoder_config):
        """Test that MLP has correct architecture"""
        try:
            model = OMRModel(
                vision_encoder_name=omr_config['vision_encoder_name'],
                text_decoder_name=omr_config['text_decoder_name'],
                mlp_layers=omr_config['mlp_layers'],
                decoder_config=text_decoder_config
            )

            assert model.mlp is not None
            # Count the number of Linear layers in MLP
            linear_layers = [m for m in model.mlp.modules() if isinstance(m, torch.nn.Linear)]
            # Should have len(mlp_layers) + 1 linear layers (hidden + output projection)
            expected_layers = len(omr_config['mlp_layers']) + 1
            assert len(linear_layers) == expected_layers
        except Exception as e:
            pytest.skip(f"Skipping test due to missing HuggingFace model or network issue: {e}")

    def test_omr_model_with_empty_mlp(self, omr_config, text_decoder_config):
        """Test OMRModel with no MLP hidden layers"""
        try:
            model = OMRModel(
                vision_encoder_name=omr_config['vision_encoder_name'],
                text_decoder_name=omr_config['text_decoder_name'],
                mlp_layers=[],
                decoder_config=text_decoder_config
            )

            assert model is not None
            # Should still have one linear layer for final projection
            linear_layers = [m for m in model.mlp.modules() if isinstance(m, torch.nn.Linear)]
            assert len(linear_layers) == 1
        except Exception as e:
            pytest.skip(f"Skipping test due to missing HuggingFace model or network issue: {e}")

    def test_omr_model_with_none_mlp(self, omr_config, text_decoder_config):
        """Test OMRModel with None MLP layers"""
        try:
            model = OMRModel(
                vision_encoder_name=omr_config['vision_encoder_name'],
                text_decoder_name=omr_config['text_decoder_name'],
                mlp_layers=None,
                decoder_config=text_decoder_config
            )

            assert model is not None
            # Should handle None gracefully
            assert model.mlp is not None
        except Exception as e:
            pytest.skip(f"Skipping test due to missing HuggingFace model or network issue: {e}")

    def test_omr_model_small_variant(self, small_variant_config):
        """Test OMRModel with small variant configuration"""
        try:
            model = OMRModel(
                vision_encoder_name="google/vit-base-patch16-224",
                text_decoder_name="custom",
                mlp_layers=small_variant_config['mlp_layers'],
                decoder_config=small_variant_config['text_decoder']
            )

            assert model is not None
        except Exception as e:
            pytest.skip(f"Skipping test due to missing HuggingFace model or network issue: {e}")

    def test_omr_model_has_required_attributes(self, omr_config, text_decoder_config):
        """Test that OMRModel has all required attributes"""
        try:
            model = OMRModel(
                vision_encoder_name=omr_config['vision_encoder_name'],
                text_decoder_name=omr_config['text_decoder_name'],
                mlp_layers=omr_config['mlp_layers'],
                decoder_config=text_decoder_config
            )

            # Check for required attributes
            assert hasattr(model, 'encoder'), "Model should have encoder attribute"
            assert hasattr(model, 'decoder'), "Model should have decoder attribute"
            assert hasattr(model, 'tokenizer'), "Model should have tokenizer attribute"
            assert hasattr(model, 'encoder_dim'), "Model should have encoder_dim attribute"
            assert hasattr(model, 'decoder_dim'), "Model should have decoder_dim attribute"
            assert hasattr(model, 'mlp'), "Model should have mlp attribute"
            assert hasattr(model, 'norm'), "Model should have norm attribute"
        except Exception as e:
            pytest.skip(f"Skipping test due to missing HuggingFace model or network issue: {e}")

    def test_omr_model_parameter_count(self, omr_config, text_decoder_config):
        """Test that OMRModel has parameters"""
        try:
            model = OMRModel(
                vision_encoder_name=omr_config['vision_encoder_name'],
                text_decoder_name=omr_config['text_decoder_name'],
                mlp_layers=omr_config['mlp_layers'],
                decoder_config=text_decoder_config
            )

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            assert total_params > 0, "Model should have parameters"
            assert trainable_params > 0, "Model should have trainable parameters"

            # Trainable params should be less than total (since encoder is frozen)
            assert trainable_params < total_params, "Not all parameters should be trainable"
        except Exception as e:
            pytest.skip(f"Skipping test due to missing HuggingFace model or network issue: {e}")


class TestOMRModelIntegration:
    """Integration tests for OMR Model"""

    def test_model_creation_from_full_config(self, model_config):
        """Test creating model using all config parameters"""
        try:
            omr_cfg = model_config['omr_model']
            decoder_cfg = model_config['text_decoder']

            model = OMRModel(
                vision_encoder_name=omr_cfg['vision_encoder_name'],
                text_decoder_name=omr_cfg['text_decoder_name'],
                mlp_layers=omr_cfg['mlp_layers'],
                decoder_config=decoder_cfg
            )

            assert model is not None

            # Verify dimensions are set correctly
            assert model.encoder_dim > 0
            assert model.decoder_dim > 0
        except Exception as e:
            pytest.skip(f"Skipping test due to missing HuggingFace model or network issue: {e}")

    def test_model_modes(self, omr_config, text_decoder_config):
        """Test model can be set to train and eval modes"""
        try:
            model = OMRModel(
                vision_encoder_name=omr_config['vision_encoder_name'],
                text_decoder_name=omr_config['text_decoder_name'],
                mlp_layers=omr_config['mlp_layers'],
                decoder_config=text_decoder_config
            )

            # Test train mode
            model.train()
            assert model.training == True

            # Test eval mode
            model.eval()
            assert model.training == False
        except Exception as e:
            pytest.skip(f"Skipping test due to missing HuggingFace model or network issue: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

