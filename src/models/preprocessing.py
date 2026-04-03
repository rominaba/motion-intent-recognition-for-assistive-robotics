from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def _resolve_data_root(data_root: str | Path) -> Path:
    return Path(data_root).expanduser().resolve()


def load_data(data_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = _resolve_data_root(data_root)
    train_dir = root / "train"
    test_dir = root / "test"
    X_train = pd.read_csv(train_dir / "X_train.txt", sep=r"\s+", header=None)
    y_train = pd.read_csv(train_dir / "y_train.txt", sep=r"\s+", header=None, names=["activity"])
    X_test = pd.read_csv(test_dir / "X_test.txt", sep=r"\s+", header=None)
    y_test = pd.read_csv(test_dir / "y_test.txt", sep=r"\s+", header=None, names=["activity"])
    return X_train, y_train, X_test, y_test


def load_target_label_set(data_root: str | Path) -> pd.DataFrame:
    root = _resolve_data_root(data_root)
    return pd.read_csv(
        root / "activity_labels.txt",
        sep=r"\s+",
        header=None,
        names=["index", "activity_name"],
    )

def target_names_for_encoder(activity_labels: pd.DataFrame, le: LabelEncoder) -> list[str]:
    return (
        activity_labels.set_index("index")
        .loc[le.classes_, "activity_name"]
        .astype(str)
        .tolist()
    )


def process_features_labels(
        X_train: pd.DataFrame, y_train: pd.DataFrame, 
        X_test: pd.DataFrame, y_test: pd.DataFrame,
        activity_labels: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train["activity"].to_numpy())
    y_test_enc = le.transform(y_test["activity"].to_numpy())
    target_names = target_names_for_encoder(activity_labels, le)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)



    return X_train_s, y_train_enc, X_test_s, y_test_enc, target_names