import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from trustcredit.data.loader import read_csv_encoded, load_dataset, prepare_datasets

def test_read_csv_encoded():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple CSV file
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        csv_path = os.path.join(tmpdir, "test.csv")
        df.to_csv(csv_path, index=False)
        
        # Test reading
        df_read = read_csv_encoded(tmpdir, "test.csv")
        pd.testing.assert_frame_equal(df, df_read)

def test_load_dataset_invalid():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("nonexistent_dataset")

def test_load_dataset_german(tmp_path):
    # Setup mock german dataset file
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    
    # Create mock German dataset
    mock_data = pd.DataFrame({
        "CheckingAccount": ["A11", "A12"],
        "Duration": [6, 48],
        "CreditHistory": ["A34", "A32"],
        "Purpose": ["A43", "A40"],
        "CreditAmount": [1169, 5951],
        "SavingsAccount": ["A65", "A61"],
        "EmploymentSince": ["A75", "A73"],
        "InstallmentRate": [4, 2],
        "Gender": ["Male", "Female"],
        "OtherDebtors": ["No", "No"],
        "ResidenceSince": [4, 2],
        "Property": ["Real estate", "Real estate"],
        "Age": [67, 22],
        "OtherInstallmentPlans": ["No", "No"],
        "Housing": ["Own", "Own"],
        "ExistingCredits": [2, 1],
        "Job": ["Skilled", "Skilled"],
        "Dependents": [1, 1],
        "Telephone": [1, 0],
        "ForeignWorker": [1, 1],
        "DEFAULT": [0, 1]
    })
    
    csv_file = prepared_dir / "german.csv"
    mock_data.to_csv(csv_file, index=False)
    
    # Load and check categorical conversion
    df = load_dataset("german", data_path=str(prepared_dir))
    
    assert isinstance(df, pd.DataFrame)
    assert df["Gender"].dtype.name == "category"
    assert df["DEFAULT"].tolist() == [0, 1]
