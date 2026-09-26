"""
Predict room occupancy from a single sensor reading using a trained model.

Usage:
    python src/predict.py --model models/model.joblib --temperature 23.2 \
        --humidity 27.3 --light 450 --co2 720 --humidity-ratio 0.0048
"""
import argparse

import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--humidity", type=float, required=True)
    parser.add_argument("--light", type=float, required=True)
    parser.add_argument("--co2", type=float, required=True)
    parser.add_argument("--humidity-ratio", type=float, required=True)
    args = parser.parse_args()

    model = joblib.load(args.model)
    row = pd.DataFrame(
        [
            {
                "Temperature": args.temperature,
                "Humidity": args.humidity,
                "Light": args.light,
                "CO2": args.co2,
                "HumidityRatio": args.humidity_ratio,
            }
        ]
    )
    prediction = int(model.predict(row)[0])
    label = "occupied" if prediction == 1 else "not occupied"
    print(f"Prediction: {label} ({prediction})")


if __name__ == "__main__":
    main()
