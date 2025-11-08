# OMR Model Tests

This directory contains unit tests and integration tests for the Optical Music Recognition models.

## Test Structure

- `test_text_decoder.py` - Tests for the Custom Text Decoder model and its components
- `test_omr_model.py` - Tests for the main OMR model

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_text_decoder.py
pytest tests/test_omr_model.py
```

### Run with verbose output
```bash
pytest -v
```

### Run with coverage report
```bash
pytest --cov=src --cov-report=html
```

### Run specific test class or function
```bash
pytest tests/test_text_decoder.py::TestCustomTextDecoder
pytest tests/test_text_decoder.py::TestCustomTextDecoder::test_decoder_creation_with_config
```

## Test Configuration

Tests use the configuration file `configs/model_config.yaml` to instantiate models with proper parameters.

## Requirements

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Notes

- OMR model tests require internet connection to download pretrained vision encoders from HuggingFace
- Tests will be skipped if models cannot be downloaded
- Text decoder tests can run offline

