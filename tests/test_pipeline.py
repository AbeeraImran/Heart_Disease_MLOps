import pandas as pd
import os


def test_data_loading():
    """Test if the dataset exists and has data."""
    file_path = "data/heart.csv"
    assert os.path.exists(file_path), "Dataset file not found!"

    df = pd.read_csv(file_path)
    assert not df.empty, "Dataset is empty!"
    assert "target" in df.columns, "Target column missing!"


def test_model_output_exists():
    """Check if the models folder is ready for artifacts."""
    assert os.path.exists("models/"), "Models directory missing!"