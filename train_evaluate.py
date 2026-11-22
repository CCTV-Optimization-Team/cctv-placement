
import os
import csv
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr

import config
from map import build_map


# --------------------------------------------------
# STEP 1: LOAD DATASET
# --------------------------------------------------

def load_dataset():

    dataset_path = os.path.join(
        "datasets",
        "all_maps_dataset.npz"
    )

    data = np.load(dataset_path, allow_pickle=True)

    X = data["X"]
    y = data["y"]
    map_names = data["map_names"]
    coordinates = data["coordinates"]

    if len(X) != len(y) or len(X) != len(map_names):
        raise ValueError("Dataset dimensions do not match!")

    if len(X) != len(coordinates):
        raise ValueError("Coordinate dimensions do not match!")

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Dataset contains invalid values!")

    print("\nDATASET LOADED SUCCESSFULLY")
    print("---------------------------")
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)
    print("Number of maps:", len(np.unique(map_names)))

    return X, y, map_names, coordinates


# --------------------------------------------------
# STEP 2: EVALUATION METRICS
# --------------------------------------------------

def evaluate_model(y_true, y_pred):

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # Spearman correlation
    if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        spearman = np.nan
    else:
        spearman = spearmanr(y_true, y_pred)[0]

    return r2, mae, spearman


# --------------------------------------------------
# STEP 3: SAVE PREDICTED PRIORITY MAP
# --------------------------------------------------

def save_prediction_map(map_name, coordinates, predictions, model_name):

    gmap = build_map(map_name)

    # Initially, every cell has priority 0.
    # Walls and unpredicted cells will remain 0.
    predicted_map = np.zeros(gmap.grid.shape, dtype=float)

    for (row, col), value in zip(coordinates, predictions):
        predicted_map[int(row), int(col)] = value

    os.makedirs(config.PRIORITY_DIR, exist_ok=True)

    file_path = os.path.join(
        config.PRIORITY_DIR,
        f"{map_name}_pred_{model_name}.npy"
    )

    np.save(file_path, predicted_map)

    print(f"Saved prediction: {file_path}")


# --------------------------------------------------
# STEP 4: LEAVE-ONE-MAP-OUT VALIDATION
# --------------------------------------------------

def run_lomo(X, y, map_names, coordinates):

    unique_maps = np.unique(map_names)

    results = []

    for held_out_map in unique_maps:

        print("\n" + "=" * 50)
        print("HELD-OUT MAP:", held_out_map)
        print("=" * 50)

        # Separate training and testing samples
        test_mask = map_names == held_out_map
        train_mask = ~test_mask

        X_train = X[train_mask]
        y_train = y[train_mask]

        X_test = X[test_mask]
        y_test = y[test_mask]

        test_coordinates = coordinates[test_mask]

        print("Training samples:", len(X_train))
        print("Testing samples:", len(X_test))

        # ------------------------------------------
        # MODEL 1: LINEAR REGRESSION
        # ------------------------------------------

        lr_model = LinearRegression()

        lr_model.fit(X_train, y_train)

        lr_predictions = lr_model.predict(X_test)

        # Priority cannot be negative
        lr_predictions = np.clip(lr_predictions, 0, None)

        lr_metrics = evaluate_model(y_test, lr_predictions)

        # ------------------------------------------
        # MODEL 2: GRADIENT BOOSTING
        # ------------------------------------------

        gb_model = GradientBoostingRegressor(
            random_state=42
        )

        gb_model.fit(X_train, y_train)

        gb_predictions = gb_model.predict(X_test)

        # Priority cannot be negative
        gb_predictions = np.clip(gb_predictions, 0, None)

        gb_metrics = evaluate_model(y_test, gb_predictions)

        # ------------------------------------------
        # BASELINE: MEAN PRIORITY
        # ------------------------------------------

        mean_value = np.mean(y_train)

        mean_predictions = np.full(
            len(y_test),
            mean_value
        )

        mean_metrics = evaluate_model(y_test, mean_predictions)

        # ------------------------------------------
        # DISPLAY RESULTS
        # ------------------------------------------

        models = [
            ("Linear Regression", lr_metrics),
            ("Gradient Boosting", gb_metrics),
            ("Mean Baseline", mean_metrics)
        ]

        for model_name, metrics in models:

            r2, mae, spearman = metrics

            print(f"\n{model_name}")
            print(f"R2 Score: {r2:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"Spearman Correlation: {spearman:.4f}")

            results.append({
                "map": str(held_out_map),
                "model": model_name,
                "R2": r2,
                "MAE": mae,
                "Spearman": spearman
            })

        # ------------------------------------------
        # SAVE PREDICTED MAPS
        # ------------------------------------------

        save_prediction_map(
            held_out_map,
            test_coordinates,
            lr_predictions,
            "lr"
        )

        save_prediction_map(
            held_out_map,
            test_coordinates,
            gb_predictions,
            "gb"
        )

    return results


# --------------------------------------------------
# STEP 5: SAVE RESULTS
# --------------------------------------------------

def save_results(results):

    os.makedirs("results", exist_ok=True)

    output_path = os.path.join(
        "results",
        "lomo_metrics.csv"
    )

    with open(output_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["map", "model", "R2", "MAE", "Spearman"]
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nMetrics saved to:", output_path)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    X, y, map_names, coordinates = load_dataset()

    results = run_lomo(
        X,
        y,
        map_names,
        coordinates
    )

    save_results(results)

    print("\nALL LOMO EXPERIMENTS COMPLETED!")
