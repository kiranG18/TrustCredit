"""
Data loading and preprocessing utilities for TrustCredit.

Supports three benchmark credit datasets:
- German Credit
- Taiwan Credit Card Default
- Home Credit Default Risk
"""

import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd


def read_csv_encoded(path: Union[str, Path], filename: str) -> pd.DataFrame:
    """Read a CSV file with automatic character encoding detection.

    Parameters
    ----------
    path : str or Path
        Directory where the file is located.
    filename : str
        Name of the CSV file to read.

    Returns
    -------
    pd.DataFrame
        DataFrame loaded from the CSV file.
    """
    path = Path(path)
    the_file = path / filename
    if not the_file.exists():
        raise FileNotFoundError(f"File not found: '{the_file}'")
    try:
        data = pd.read_csv(the_file, index_col=False)
    except UnicodeDecodeError:
        import chardet
        rawdata = the_file.read_bytes()
        result = chardet.detect(rawdata)
        charenc = result["encoding"]
        data = pd.read_csv(the_file, encoding=charenc, index_col=False)
    return data


def download_datasets(output_dir: Union[str, Path] = ".") -> None:
    """Download all benchmark datasets from Google Drive and unzip them.

    Parameters
    ----------
    output_dir : str or Path, optional
        Directory to download and extract data into, by default "."
    """
    import gdown
    import zipfile
    output_dir = Path(output_dir)
    url = "https://drive.google.com/uc?id=1Y7bTNsxDv-te40FnJsoca1YeB4da6TCq"
    output = output_dir / "data.zip"
    gdown.download(url, str(output), quiet=False)
    with zipfile.ZipFile(output, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    output.unlink()


def prepare_datasets(data_path: Union[str, Path] = "data") -> None:
    """Preprocess raw datasets and save them as cleaned CSVs.

    Reads raw files from `data_path`, applies column renaming, type casting,
    and categorical encoding, and saves prepared CSVs to `data_path/prepared/`.

    Parameters
    ----------
    data_path : str or Path, optional
        Root folder containing raw dataset subdirectories, by default "data"
    """
    data_path = Path(data_path)
    prepared_dir = data_path / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    # --- Home Credit ---
    home_credit_dir = data_path / "HomeCredit"
    df = read_csv_encoded(home_credit_dir, "application_train.csv")
    df = df.drop(columns=["SK_ID_CURR", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"])
    df = df.rename(columns={"TARGET": "DEFAULT"})
    for col in df.select_dtypes("object").columns:
        df[col] = pd.Categorical(df[col])
    df.to_csv(prepared_dir / "homecredit.csv", index=False)

    # --- Taiwan ---
    taiwan_dir = data_path / "Taiwan"
    df = read_csv_encoded(taiwan_dir, "Taiwan.csv")
    df.columns = df.iloc[0, :].tolist()
    df = df.iloc[1:, :].drop(columns=["ID"])
    df = df.rename(columns={"default payment next month": "DEFAULT"}).astype("float64")
    df["SEX"] = df["SEX"].map({2: "Female", 1: "Male"})
    df["EDUCATION"] = df["EDUCATION"].map({
        -2: "Unknown", -1: "Unknown", 0: "Unknown",
        1: "Graduate School", 2: "University", 3: "High School",
        4: "Others", 5: "Unknown", 6: "Unknown",
    })
    df["MARRIAGE"] = df["MARRIAGE"].map({0: "Others", 1: "Married", 2: "Single", 3: "Others"})
    cat_cols = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    for col in cat_cols:
        df[col] = pd.Categorical(df[col])
    df.to_csv(prepared_dir / "taiwan.csv", index=False)

    # --- German Credit ---
    german_dir = data_path / "German"
    df = read_csv_encoded(german_dir, "german.csv")
    df.columns = [
        "CheckingAccount", "Duration", "CreditHistory", "Purpose", "CreditAmount",
        "SavingsAccount", "EmploymentSince", "InstallmentRate", "PersonalStatus",
        "OtherDebtors", "ResidenceSince", "Property", "Age", "OtherInstallmentPlans",
        "Housing", "ExistingCredits", "Job", "Dependents", "Telephone", "ForeignWorker", "DEFAULT",
    ]
    df["DEFAULT"] = df["DEFAULT"].apply(lambda x: 1 if x == 2 else 0)
    df["Gender"] = df["PersonalStatus"].map(
        {"A91": "Male", "A92": "Female", "A93": "Male", "A94": "Male", "A95": "Female"}
    )
    df["CheckingAccount"] = df["CheckingAccount"].map(
        {"A11": "< 0", "A12": "0 - 200", "A13": "> 200", "A14": "No"}
    )
    df["CreditHistory"] = df["CreditHistory"].map({
        "A30": "No credits/all paid", "A31": "All paid", "A32": "Existing paid",
        "A33": "Delay in paying", "A34": "Critical account",
    })
    df["Purpose"] = df["Purpose"].map({
        "A40": "Car (new)", "A41": "Car (used)", "A42": "Furniture/equipment",
        "A43": "Radio/television", "A44": "Domestic appliances", "A45": "Repairs",
        "A46": "Education", "A47": "Vacation", "A48": "Retraining",
        "A49": "Business", "A410": "Others",
    })
    df["SavingsAccount"] = df["SavingsAccount"].map(
        {"A61": "< 100", "A62": "100 - 500", "A63": "500 - 1000", "A64": "> 1000", "A65": "Unknown/None"}
    )
    df["EmploymentSince"] = df["EmploymentSince"].map(
        {"A71": "Unemployed", "A72": "< 1", "A73": "1 - 4", "A74": "4 - 7", "A75": "> 7"}
    )
    df["OtherDebtors"] = df["OtherDebtors"].map({"A101": "No", "A102": "Co-applicant", "A103": "Guarantor"})
    df["Property"] = df["Property"].map({
        "A121": "Real estate", "A122": "Savings agreement/life insurance",
        "A123": "Car or other", "A124": "Unknown/None",
    })
    df["OtherInstallmentPlans"] = df["OtherInstallmentPlans"].map({"A141": "Bank", "A142": "Stores", "A143": "No"})
    df["Housing"] = df["Housing"].map({"A151": "Rent", "A152": "Own", "A153": "For free"})
    df["Job"] = df["Job"].map(
        {"A171": "Unemployed", "A172": "Unskilled", "A173": "Skilled", "A174": "Highly skilled"}
    )
    df["Telephone"] = df["Telephone"].map({"A191": 0, "A192": 1})
    df["ForeignWorker"] = df["ForeignWorker"].map({"A201": 1, "A202": 0})
    df = df.drop(columns=["PersonalStatus"])
    cat_cols = [
        "CheckingAccount", "CreditHistory", "Purpose", "SavingsAccount", "EmploymentSince",
        "Gender", "OtherDebtors", "Property", "OtherInstallmentPlans", "Housing", "Job",
        "Telephone", "ForeignWorker",
    ]
    for col in cat_cols:
        df[col] = pd.Categorical(df[col])
    df.to_csv(prepared_dir / "german.csv", index=False)
    print(f"All datasets prepared and saved to {prepared_dir}")


def load_dataset(dataset_name: str, data_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load a prepared benchmark credit dataset by name.

    Parameters
    ----------
    dataset_name : str
        One of "homecredit", "taiwan", or "german".
    data_path : str or Path, optional
        Root path to the data directory. If None, defaults to "data/prepared/".

    Returns
    -------
    pd.DataFrame
        DataFrame with categorical columns restored to pd.Categorical dtype.

    Raises
    ------
    ValueError
        If `dataset_name` is not one of the supported datasets.
    FileNotFoundError
        If the prepared dataset file cannot be found.
    """
    if data_path is None:
        data_path = Path("data/prepared")
    else:
        data_path = Path(data_path)

    dataset_configs = {
        "homecredit": {
            "file": "homecredit.csv",
            "cat_cols": "infer",
        },
        "taiwan": {
            "file": "taiwan.csv",
            "cat_cols": ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"],
        },
        "german": {
            "file": "german.csv",
            "cat_cols": [
                "CheckingAccount", "CreditHistory", "Purpose", "SavingsAccount", "EmploymentSince",
                "Gender", "OtherDebtors", "Property", "OtherInstallmentPlans", "Housing", "Job",
                "Telephone", "ForeignWorker",
            ],
        },
    }

    if dataset_name not in dataset_configs:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Choose from: {list(dataset_configs.keys())}"
        )

    config = dataset_configs[dataset_name]
    target_file = data_path / config["file"]
    
    if not target_file.exists():
        raise FileNotFoundError(
            f"Dataset file not found: '{target_file}'. "
            "Please ensure you run `prepare_datasets()` or the download script first."
        )

    df = pd.read_csv(target_file)

    cat_cols = config["cat_cols"]
    if cat_cols == "infer":
        cat_cols = df.select_dtypes("object").columns.tolist()

    for col in cat_cols:
        df[col] = pd.Categorical(df[col])

    return df


if __name__ == "__main__":
    download_datasets()
    prepare_datasets()
