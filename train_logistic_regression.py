from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.models.logistic_regression import LogisticRegressionModel
from src.models.pca_reduction import fit_best_pca_then_transform
from src.models.preprocessing import load_data, load_target_label_set, process_features_labels
from src.utils import DATA_DIR, PROJECT_ROOT, evaluate_model, get_logger, set_seed

logger = get_logger("logistic_regression", write_to_file=True)
optuna.logging.set_verbosity(optuna.logging.INFO)


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
        help="Maximum training iterations (ignored when --n-trials > 0)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Gradient descent learning rate (ignored when --n-trials > 0)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=0,
        help="Optuna trials; 0 disables tuning and uses --learning-rate / --max-iter",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Holdout fraction for --pca (n_components search) and for --n-trials > 0 (Optuna)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / f"logistic-regression-{datetime.now().strftime('%Y-%m-%d-%H-%M')}.npz"),
        help="Path to write trained weights and metadata (.npz)",
    )
    parser.add_argument(
        "--pca",
        action="store_true",
        help="If set, search PCA n_components on a val split, then reduce features before training.",
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

    if args.pca:
        X_train, X_test, _, pca_meta = fit_best_pca_then_transform(
            X_train,
            y_train,
            X_test,
            random_state=args.random_state,
            val_fraction=args.val_fraction,
            logger=logger,
        )
        logger.info(
            "After PCA: X_train %s, X_test %s (n_components=%s)",
            X_train.shape,
            X_test.shape,
            pca_meta["n_components"],
        )

    learning_rate = args.learning_rate
    max_iter = args.max_iter
    if args.n_trials > 0:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=args.val_fraction,
            random_state=args.random_state,
            stratify=y_train,
        )

        def objective(trial: optuna.Trial) -> float:
            lr = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True)
            n_iter = trial.suggest_int("max_iter", 500, 10000)

            m = LogisticRegressionModel(random_state=args.random_state, max_iter=n_iter)
            m.fit(X_tr, y_tr, learning_rate=lr)

            val_acc = float(accuracy_score(y_val, m.predict(X_val)))

            # Small regularization against inefficient hyperparameters.
            # Keep accuracy as the main target, but break ties in favor of
            # larger learning rates and fewer iterations to improve efficiency and prevent overfitting
            iter_penalty = 0.01 * (n_iter / 10000)
            lr_penalty = 0.01 * max(0.0, np.log10(1e-2 / lr))

            score = val_acc - iter_penalty - lr_penalty

            trial.set_user_attr("val_accuracy", val_acc)
            return score

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=args.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0),
        )
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

        learning_rate = study.best_params["learning_rate"]
        max_iter = study.best_params["max_iter"]

        logger.info(
            "Optuna best params: %s (penalized score %.4f, val accuracy %.4f)",
            study.best_params,
            study.best_value,
            study.best_trial.user_attrs["val_accuracy"],
        )

    model = LogisticRegressionModel(random_state=args.random_state, max_iter=max_iter)
    logger.info("Training model...")
    model.fit(X_train, y_train, learning_rate=learning_rate)
    logger.info("Training completed")

    logger.info("Predicting train set:")
    pred = model.predict(X_train)
    evaluate_model(y_true=y_train, y_pred=pred, target_names=target_names, logger=logger)

    logger.info("Predicting test set:")
    pred = model.predict(X_test)
    evaluate_model(y_true=y_test, y_pred=pred, target_names=target_names, logger=logger)
    
    if args.output_path:
        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            weight_matrix=model.weight_matrix,
            bias_vector=model.bias_vector,
            max_iter=np.int64(model.max_iter),
            random_state=np.int64(model.random_state),
            learning_rate=np.float64(learning_rate),
            target_names=np.array(target_names, dtype=object),
        )
        logger.info("Saved model to %s", out.resolve())


if __name__ == "__main__":
    main()
