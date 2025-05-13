#!/usr/bin/env python3
"""Training script for California Housing model."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from california_housing.preprocessing.preprocessor import HousingPreprocessor, prepare_data
from california_housing.utils.evaluation import evaluate_model, format_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train California Housing model')
    parser.add_argument('--config', type=str, default='config.json',
                      help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='models',
                      help='Directory to save model and metrics')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load config if exists, else use defaults
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Config file {args.config} not found, using defaults")
        config = {
            'random_state': 42,
            'test_size': 0.2,
            'handle_outliers': True,
            'handle_skewness': True,
            'model_params': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2
            }
        }
    
    # Load data
    print("Loading data...")
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    
    # Prepare features and target
    X, y = prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['test_size'],
        random_state=config['random_state']
    )
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessor = HousingPreprocessor(
        handle_outliers=config['handle_outliers'],
        handle_skewness=config['handle_skewness']
    )
    X_train_processed, y_train = preprocessor.fit_transform(X_train, y_train)
    X_test_processed, y_test = preprocessor.transform(X_test, y_test)
    
    # Train model
    print("Training model...")
    model = RandomForestRegressor(
        random_state=config['random_state'],
        **config['model_params']
    )
    model.fit(X_train_processed, y_train)
    
    # Evaluate
    print("Evaluating model...")
    train_metrics = evaluate_model(model, X_train_processed, y_train)
    test_metrics = evaluate_model(model, X_test_processed, y_test)
    
    metrics = pd.DataFrame({
        'Train': format_metrics(train_metrics),
        'Test': format_metrics(test_metrics)
    })
    print("\nModel Performance:")
    print(metrics)
    
    # Save artifacts
    print("\nSaving artifacts...")
    joblib.dump(preprocessor, output_dir / 'preprocessor.joblib')
    joblib.dump(model, output_dir / 'model.joblib')
    metrics.to_csv(output_dir / 'metrics.csv')
    
    print(f"\nArtifacts saved to {output_dir}")


if __name__ == '__main__':
    main()
