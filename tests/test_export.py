"""
Tests for the data export module.
"""

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from src.data_preparation import export


@pytest.fixture
def sample_file_content():
    """Sample file content for testing."""
    return "Sample file content for testing."


@pytest.fixture
def sample_file(sample_file_content, tmp_path):
    """Create a sample file for testing."""
    file_path = tmp_path / "sample.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(sample_file_content)
    return file_path


@pytest.fixture
def sample_jsonl_content():
    """Sample JSONL content for testing."""
    return [
        {"input_text": "What is machine learning?", "output_text": "Machine learning is a branch of artificial intelligence that focuses on developing systems that can learn from data."},
        {"input_text": "Explain neural networks", "output_text": "Neural networks are computing systems inspired by the biological neural networks that constitute animal brains."},
        {"input_text": "What is deep learning?", "output_text": "Deep learning is a subset of machine learning that uses neural networks with multiple layers."},
    ]


@pytest.fixture
def sample_data_dir(sample_jsonl_content, tmp_path):
    """Create a sample data directory with train, val, and test files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create train, val, and test files
    splits = {
        "train": sample_jsonl_content[:2],
        "val": sample_jsonl_content[2:3],
        "test": sample_jsonl_content[1:2],
    }
    
    for split, content in splits.items():
        file_path = data_dir / f"{split}.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for item in content:
                f.write(json.dumps(item) + "\n")
    
    return data_dir


@pytest.fixture
def mock_storage_client():
    """Mock Google Cloud Storage client."""
    with mock.patch("google.cloud.storage.Client") as mock_client:
        # Configure the mock
        mock_bucket = mock.MagicMock()
        mock_blob = mock.MagicMock()
        
        mock_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        yield mock_client


def test_export_to_gcs(sample_file, mock_storage_client):
    """Test exporting a file to GCS."""
    # Export file
    gcs_uri = export.export_to_gcs(
        local_path=sample_file,
        gcs_path="test/sample.txt",
        project_id="test-project",
        bucket_name="test-bucket",
    )
    
    # Check that GCS client was called correctly
    mock_storage_client.assert_called_once_with(project="test-project")
    mock_storage_client.return_value.bucket.assert_called_once_with("test-bucket")
    mock_storage_client.return_value.bucket.return_value.blob.assert_called_once_with("test/sample.txt")
    mock_storage_client.return_value.bucket.return_value.blob.return_value.upload_from_filename.assert_called_once_with(str(sample_file))
    
    # Check that GCS URI was returned
    assert gcs_uri == "gs://test-bucket/test/sample.txt"


def test_export_directory_to_gcs(tmp_path, mock_storage_client):
    """Test exporting a directory to GCS."""
    # Create test directory with files
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    
    for i in range(3):
        file_path = test_dir / f"file_{i}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Content of file {i}")
    
    # Export directory
    gcs_uris = export.export_directory_to_gcs(
        local_dir=test_dir,
        gcs_dir="test/dir",
        project_id="test-project",
        bucket_name="test-bucket",
        file_pattern="*.txt",
        recursive=False,
    )
    
    # Check that GCS client was called correctly
    assert mock_storage_client.call_count == 1
    assert mock_storage_client.return_value.bucket.call_count == 3
    assert mock_storage_client.return_value.bucket.return_value.blob.call_count == 3
    assert mock_storage_client.return_value.bucket.return_value.blob.return_value.upload_from_filename.call_count == 3
    
    # Check that GCS URIs were returned
    assert len(gcs_uris) == 3
    for uri in gcs_uris:
        assert uri.startswith("gs://test-bucket/test/dir/")
        assert uri.endswith(".txt")


def test_export_training_data_to_gcs(sample_data_dir, mock_storage_client):
    """Test exporting training data to GCS."""
    # Export training data
    gcs_uris = export.export_training_data_to_gcs(
        data_dir=sample_data_dir,
        project_id="test-project",
        bucket_name="test-bucket",
        gcs_dir="training_data",
    )
    
    # Check that GCS client was called correctly
    assert mock_storage_client.call_count == 1
    assert mock_storage_client.return_value.bucket.call_count == 3  # train, val, test
    assert mock_storage_client.return_value.bucket.return_value.blob.call_count == 3
    assert mock_storage_client.return_value.bucket.return_value.blob.return_value.upload_from_filename.call_count == 3
    
    # Check that GCS URIs were returned
    assert "train" in gcs_uris
    assert "val" in gcs_uris
    assert "test" in gcs_uris
    
    assert gcs_uris["train"][0] == "gs://test-bucket/training_data/train.jsonl"
    assert gcs_uris["val"][0] == "gs://test-bucket/training_data/val.jsonl"
    assert gcs_uris["test"][0] == "gs://test-bucket/training_data/test.jsonl"


def test_export_metadata_to_gcs(tmp_path, mock_storage_client):
    """Test exporting metadata to GCS."""
    # Create test metadata directory with files
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    
    for i in range(3):
        file_path = metadata_dir / f"doc_{i}_metadata.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"title": f"Document {i}"}, f)
    
    # Export metadata
    gcs_uris = export.export_metadata_to_gcs(
        metadata_dir=metadata_dir,
        project_id="test-project",
        bucket_name="test-bucket",
        gcs_dir="metadata",
    )
    
    # Check that GCS client was called correctly
    assert mock_storage_client.call_count == 1
    assert mock_storage_client.return_value.bucket.call_count == 3
    assert mock_storage_client.return_value.bucket.return_value.blob.call_count == 3
    assert mock_storage_client.return_value.bucket.return_value.blob.return_value.upload_from_filename.call_count == 3
    
    # Check that GCS URIs were returned
    assert len(gcs_uris) == 3
    for uri in gcs_uris:
        assert uri.startswith("gs://test-bucket/metadata/")
        assert uri.endswith(".json")


def test_create_manifest_file(tmp_path):
    """Test creating a manifest file."""
    # Create test GCS URIs
    gcs_uris = {
        "train": ["gs://test-bucket/training_data/train.jsonl"],
        "val": ["gs://test-bucket/training_data/val.jsonl"],
        "test": ["gs://test-bucket/training_data/test.jsonl"],
    }
    
    # Create manifest file
    manifest_path = export.create_manifest_file(
        gcs_uris=gcs_uris,
        output_path=tmp_path / "manifest.json",
    )
    
    # Check that manifest file was created
    assert Path(manifest_path).exists()
    
    # Check that manifest file contains expected content
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    assert "train" in manifest
    assert "val" in manifest
    assert "test" in manifest
    
    assert manifest["train"][0] == "gs://test-bucket/training_data/train.jsonl"
    assert manifest["val"][0] == "gs://test-bucket/training_data/val.jsonl"
    assert manifest["test"][0] == "gs://test-bucket/training_data/test.jsonl"


def test_export_all_to_gcs(tmp_path, mock_storage_client):
    """Test exporting all data to GCS."""
    # Create test base directory with subdirectories
    base_dir = tmp_path / "base_dir"
    base_dir.mkdir()
    
    # Create extracted_text directory
    text_dir = base_dir / "extracted_text"
    text_dir.mkdir()
    
    for i in range(2):
        file_path = text_dir / f"doc_{i}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Content of document {i}")
    
    # Create metadata directory
    metadata_dir = text_dir / "metadata"
    metadata_dir.mkdir()
    
    for i in range(2):
        file_path = metadata_dir / f"doc_{i}_metadata.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"title": f"Document {i}"}, f)
    
    # Create training_data directory
    training_data_dir = base_dir / "training_data"
    training_data_dir.mkdir()
    
    for split in ["train", "val", "test"]:
        file_path = training_data_dir / f"{split}.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write('{"input_text": "Test", "output_text": "Test"}\n')
    
    # Export all data
    result = export.export_all_to_gcs(
        base_dir=base_dir,
        project_id="test-project",
        bucket_name="test-bucket",
        gcs_base_dir="test",
    )
    
    # Check that result contains expected keys
    assert "text_files" in result
    assert "metadata_files" in result
    assert "training_data" in result
    assert "manifest" in result
    
    # Check that manifest file was created
    manifest_path = base_dir / "gcs_manifest.json"
    assert manifest_path.exists()


def test_export_to_gcs_with_invalid_path():
    """Test exporting a file to GCS with an invalid path."""
    with pytest.raises(FileNotFoundError):
        export.export_to_gcs(
            local_path="nonexistent.txt",
            gcs_path="test/nonexistent.txt",
            project_id="test-project",
            bucket_name="test-bucket",
        )


def test_export_directory_to_gcs_with_invalid_path():
    """Test exporting a directory to GCS with an invalid path."""
    with pytest.raises(FileNotFoundError):
        export.export_directory_to_gcs(
            local_dir="nonexistent_dir",
            gcs_dir="test/nonexistent_dir",
            project_id="test-project",
            bucket_name="test-bucket",
        )


def test_export_training_data_to_gcs_with_invalid_path():
    """Test exporting training data to GCS with an invalid path."""
    with pytest.raises(FileNotFoundError):
        export.export_training_data_to_gcs(
            data_dir="nonexistent_dir",
            project_id="test-project",
            bucket_name="test-bucket",
        )


def test_export_metadata_to_gcs_with_invalid_path():
    """Test exporting metadata to GCS with an invalid path."""
    with pytest.raises(FileNotFoundError):
        export.export_metadata_to_gcs(
            metadata_dir="nonexistent_dir",
            project_id="test-project",
            bucket_name="test-bucket",
        )


def test_export_all_to_gcs_with_invalid_path():
    """Test exporting all data to GCS with an invalid path."""
    with pytest.raises(FileNotFoundError):
        export.export_all_to_gcs(
            base_dir="nonexistent_dir",
            project_id="test-project",
            bucket_name="test-bucket",
        )
