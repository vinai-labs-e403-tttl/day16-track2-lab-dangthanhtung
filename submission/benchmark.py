import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, 
    precision_score, recall_score, confusion_matrix
)
import time
import json

def run_benchmark():
    results = {}
    
    # 1. Load Data
    print("--- 1. Loading Data ---")
    start_time = time.time()
    # Giả sử file csv nằm cùng thư mục sau khi unzip kaggle dataset
    df = pd.read_csv('creditcard.csv')
    load_time = time.time() - start_time
    results['load_time_sec'] = round(load_time, 4)
    print(f"Data loaded in: {results['load_time_sec']} seconds")

    # 2. Preprocessing
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Training
    print("\n--- 2. Training LightGBM ---")
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1
    }

    start_time = time.time()
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data]
    )
    train_time = time.time() - start_time
    results['training_time_sec'] = round(train_time, 4)
    results['best_iteration'] = gbm.best_iteration
    print(f"Training completed in: {results['training_time_sec']} seconds")

    # 4. Evaluation
    print("\n--- 3. Evaluating Model ---")
    y_pred_prob = gbm.predict(X_test)
    y_pred = [1 if x > 0.5 else 0 for x in y_pred_prob]

    results['auc_roc'] = round(roc_auc_score(y_test, y_pred_prob), 4)
    results['accuracy'] = round(accuracy_score(y_test, y_pred), 4)
    results['f1_score'] = round(f1_score(y_test, y_pred), 4)
    results['precision'] = round(precision_score(y_test, y_pred), 4)
    results['recall'] = round(recall_score(y_test, y_pred), 4)

    for metric, value in results.items():
        if metric not in ['load_time_sec', 'training_time_sec', 'best_iteration']:
            print(f"{metric.upper()}: {value}")

    # 5. Inference Latency (1 row)
    print("\n--- 4. Benchmarking Inference ---")
    one_row = X_test.iloc[[0]]
    
    # Warping up
    gbm.predict(one_row) 
    
    latencies = []
    for _ in range(100):
        start_lat = time.time()
        gbm.predict(one_row)
        latencies.append(time.time() - start_lat)
    
    results['inference_latency_avg_ms'] = round(np.mean(latencies) * 1000, 4)
    print(f"Average Inference Latency (1 row): {results['inference_latency_avg_ms']} ms")

    # 6. Throughput (1000 rows)
    thousand_rows = X_test.iloc[:1000]
    start_tp = time.time()
    gbm.predict(thousand_rows)
    tp_time = time.time() - start_tp
    
    results['inference_throughput_1000rows_sec'] = round(tp_time, 4)
    print(f"Throughput for 1000 rows: {results['inference_throughput_1000rows_sec']} seconds")

    # Save to JSON
    with open('benchmark_result.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to benchmark_result.json")

if __name__ == "__main__":
    run_benchmark()
