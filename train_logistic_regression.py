import argparse

from src.models.logistic_regression import LogisticRegressionModel
from src.models.preprocessing import load_data, load_target_label_set, process_features_labels
from src.utils import DATA_DIR, evaluate_model, get_logger, set_seed

logger = get_logger("logistic_regression")


def parse_args():
    parser = argparse.ArgumentParser(description="Train logistic regression on motion intent data.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DATA_DIR),
        help=f"UCI HAR-style data root (contains train/, test/, activity_labels.txt). Default: {DATA_DIR}",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for weight initialization",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=10000,
        help="Maximum training iterations",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Gradient descent learning rate",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.random_state)
    X_train, y_train, X_test, y_test = load_data(args.data_path)
    activity_labels = load_target_label_set(args.data_path)
    # Encode labels and scale features
    X_train, y_train, X_test, y_test, target_names = process_features_labels(X_train, y_train, X_test, y_test, activity_labels)
    logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
    model = LogisticRegressionModel(random_state=args.random_state, max_iter=args.max_iter)
    logger.info(f"Training model...")
    model.fit(X_train, y_train, learning_rate=args.learning_rate)
    logger.info(f"Training completed")
    logger.info(f"Predicting train set:")
    pred = model.predict(X_train)
    evaluate_model(y_true=y_train, y_pred=pred, target_names=target_names, logger=logger)

    logger.info(f"Predicting test set:")
    pred = model.predict(X_test)
    evaluate_model(y_true=y_test, y_pred=pred, target_names=target_names, logger=logger)


if __name__ == "__main__":
    main()
