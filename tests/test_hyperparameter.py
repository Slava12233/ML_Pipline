"""
Tests for the hyperparameter optimization module.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import optuna
import yaml

from src.fine_tuning.config import FineTuningConfig
from src.fine_tuning.hyperparameter import (
    define_search_space,
    create_config_from_params,
    objective,
    optimize_hyperparameters,
    finetune_with_best_params,
)


@pytest.fixture
def sample_config():
    """Sample fine-tuning configuration."""
    return FineTuningConfig(
        model_name="gemini-pro",
        vertex_ai={"project_id": "test-project"},
    )


@pytest.fixture
def sample_params():
    """Sample hyperparameters."""
    return {
        "learning_rate": 5e-5,
        "batch_size": 16,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "weight_decay": 0.05,
        "warmup_ratio": 0.1,
    }


@pytest.fixture
def mock_trial():
    """Mock Optuna trial."""
    trial = mock.MagicMock()
    
    # Configure suggest methods
    trial.suggest_float.side_effect = lambda name, low, high, **kwargs: {
        "learning_rate": 5e-5,
        "lora_dropout": 0.1,
        "weight_decay": 0.05,
        "warmup_ratio": 0.1,
    }[name]
    
    trial.suggest_categorical.side_effect = lambda name, choices: {
        "batch_size": 16,
    }[name]
    
    trial.suggest_int.side_effect = lambda name, low, high, **kwargs: {
        "lora_r": 8,
        "lora_alpha": 16,
    }[name]
    
    return trial


@pytest.fixture
def mock_finetune_model():
    """Mock finetune_model function."""
    with mock.patch("src.fine_tuning.hyperparameter.finetune_model") as mock_fn:
        # Configure mock
        mock_fn.return_value = "/tmp/model"
        yield mock_fn


@pytest.fixture
def mock_setup_training_environment():
    """Mock setup_training_environment function."""
    with mock.patch("src.fine_tuning.setup.setup_training_environment") as mock_fn:
        # Configure mock
        mock_model = mock.MagicMock()
        mock_tokenizer = mock.MagicMock()
        mock_env_config = {}
        mock_fn.return_value = (mock_model, mock_tokenizer, mock_env_config)
        yield mock_fn


@pytest.fixture
def mock_load_training_data():
    """Mock load_training_data function."""
    with mock.patch("src.fine_tuning.train.load_training_data") as mock_fn:
        # Configure mock
        mock_train_dataset = mock.MagicMock()
        mock_train_dataset.__len__.return_value = 100
        mock_val_dataset = mock.MagicMock()
        mock_fn.return_value = (mock_train_dataset, mock_val_dataset)
        yield mock_fn


def test_define_search_space(mock_trial):
    """Test defining hyperparameter search space."""
    # Define search space
    params = define_search_space(mock_trial)
    
    # Check that parameters were defined
    assert "learning_rate" in params
    assert "batch_size" in params
    assert "lora_r" in params
    assert "lora_alpha" in params
    assert "lora_dropout" in params
    assert "weight_decay" in params
    assert "warmup_ratio" in params
    
    # Check parameter values
    assert params["learning_rate"] == 5e-5
    assert params["batch_size"] == 16
    assert params["lora_r"] == 8
    assert params["lora_alpha"] == 16
    assert params["lora_dropout"] == 0.1
    assert params["weight_decay"] == 0.05
    assert params["warmup_ratio"] == 0.1


def test_create_config_from_params(sample_config, sample_params):
    """Test creating configuration from hyperparameters."""
    # Create configuration
    config = create_config_from_params(sample_config, sample_params)
    
    # Check that configuration was updated
    assert config.training.learning_rate == sample_params["learning_rate"]
    assert config.training.batch_size == sample_params["batch_size"]
    assert config.training.weight_decay == sample_params["weight_decay"]
    assert config.training.warmup_ratio == sample_params["warmup_ratio"]
    assert config.lora.r == sample_params["lora_r"]
    assert config.lora.alpha == sample_params["lora_alpha"]
    assert config.lora.dropout == sample_params["lora_dropout"]


def test_objective(
    mock_trial,
    sample_config,
    mock_finetune_model,
    mock_setup_training_environment,
    mock_load_training_data,
):
    """Test objective function."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test data directory
        data_dir = os.path.join(tmp_dir, "data")
        os.makedirs(data_dir)
        
        # Create test model directory
        model_dir = os.path.join(tmp_dir, "model")
        os.makedirs(model_dir)
        
        # Create trainer state file
        trainer_state = {
            "best_metric": 0.5,
            "log_history": [
                {"eval_loss": 0.6},
                {"eval_loss": 0.5},
            ],
        }
        
        with open(os.path.join(model_dir, "trainer_state.json"), "w") as f:
            json.dump(trainer_state, f)
        
        # Configure mock
        mock_finetune_model.return_value = model_dir
        
        # Run objective function
        metric = objective(mock_trial, data_dir, sample_config, tmp_dir)
        
        # Check that metric was returned
        assert metric == 0.5
        
        # Check that finetune_model was called
        mock_finetune_model.assert_called_once()


def test_optimize_hyperparameters(
    sample_config,
    mock_finetune_model,
    mock_setup_training_environment,
    mock_load_training_data,
):
    """Test optimizing hyperparameters."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test data directory
        data_dir = os.path.join(tmp_dir, "data")
        os.makedirs(data_dir)
        
        # Create test output directory
        output_dir = os.path.join(tmp_dir, "output")
        
        # Create trainer state file
        model_dir = os.path.join(tmp_dir, "model")
        os.makedirs(model_dir)
        
        trainer_state = {
            "best_metric": 0.5,
            "log_history": [
                {"eval_loss": 0.6},
                {"eval_loss": 0.5},
            ],
        }
        
        with open(os.path.join(model_dir, "trainer_state.json"), "w") as f:
            json.dump(trainer_state, f)
        
        # Configure mock
        mock_finetune_model.return_value = model_dir
        
        # Mock optuna.create_study
        with mock.patch("optuna.create_study") as mock_create_study:
            # Configure mock
            mock_study = mock.MagicMock()
            mock_study.best_params = {
                "learning_rate": 5e-5,
                "batch_size": 16,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "weight_decay": 0.05,
                "warmup_ratio": 0.1,
            }
            mock_create_study.return_value = mock_study
            
            # Mock optuna.save_study
            with mock.patch("optuna.save_study") as mock_save_study:
                # Optimize hyperparameters
                best_params, best_config = optimize_hyperparameters(
                    data_dir=data_dir,
                    output_dir=output_dir,
                    config=sample_config,
                    n_trials=1,
                )
                
                # Check that best parameters were returned
                assert best_params == mock_study.best_params
                
                # Check that best configuration was created
                assert best_config.training.learning_rate == best_params["learning_rate"]
                assert best_config.training.batch_size == best_params["batch_size"]
                assert best_config.lora.r == best_params["lora_r"]
                
                # Check that files were created
                assert os.path.exists(os.path.join(output_dir, "best_params.json"))
                assert os.path.exists(os.path.join(output_dir, "best_config.yaml"))
                
                # Check that study was saved
                mock_save_study.assert_called_once()


def test_finetune_with_best_params(sample_config, mock_finetune_model):
    """Test fine-tuning with best hyperparameters."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test data directory
        data_dir = os.path.join(tmp_dir, "data")
        os.makedirs(data_dir)
        
        # Create test output directory
        output_dir = os.path.join(tmp_dir, "output")
        os.makedirs(output_dir)
        
        # Create test HPO directory
        hpo_dir = os.path.join(tmp_dir, "hpo")
        os.makedirs(hpo_dir)
        
        # Create best configuration file
        config_dict = sample_config.dict()
        config_path = os.path.join(hpo_dir, "best_config.yaml")
        
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)
        
        # Configure mock
        mock_finetune_model.return_value = os.path.join(output_dir, "model")
        
        # Mock load_config
        with mock.patch("src.fine_tuning.hyperparameter.load_config") as mock_load_config:
            # Configure mock
            mock_load_config.return_value = sample_config
            
            # Fine-tune with best parameters
            model_path = finetune_with_best_params(
                data_dir=data_dir,
                output_dir=output_dir,
                hpo_dir=hpo_dir,
            )
            
            # Check that model path was returned
            assert model_path == os.path.join(output_dir, "model")
            
            # Check that finetune_model was called
            mock_finetune_model.assert_called_once_with(
                data_dir=data_dir,
                output_dir=output_dir,
                config=sample_config,
            )


def test_finetune_with_best_params_nonexistent_config():
    """Test fine-tuning with nonexistent best configuration."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test data directory
        data_dir = os.path.join(tmp_dir, "data")
        os.makedirs(data_dir)
        
        # Create test output directory
        output_dir = os.path.join(tmp_dir, "output")
        os.makedirs(output_dir)
        
        # Create test HPO directory
        hpo_dir = os.path.join(tmp_dir, "hpo")
        os.makedirs(hpo_dir)
        
        # Fine-tune with best parameters
        with pytest.raises(FileNotFoundError):
            finetune_with_best_params(
                data_dir=data_dir,
                output_dir=output_dir,
                hpo_dir=hpo_dir,
            )
