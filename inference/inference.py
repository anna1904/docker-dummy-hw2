import numpy as np
import concurrent.futures
from typing import Tuple
from tqdm import tqdm
import time
import ray
import seaborn as sns
import dask
import dask.distributed as dd
from sklearn import svm



def train_model(x_train: np.ndarray, y_train: np.ndarray) -> svm:

    model = svm.SVR()
    model.fit(x_train, y_train)
    return model


def get_data(inference_size: int = 10000000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tips_dataset = sns.load_dataset("tips")
    x_train = tips_dataset["total_bill"].values.reshape(-1, 1)
    y_train = tips_dataset["tip"].values.reshape(-1, 1)

    min_total_bill = tips_dataset["total_bill"].min()
    max_total_bill = tips_dataset["total_bill"].max()
    x_test = np.random.uniform(low=min_total_bill, high=max_total_bill, size=(inference_size, 1))

    return x_train, y_train, x_test


def predict(model: svm, x: np.ndarray) -> np.ndarray:
    return model.predict(x)

#
def run_inference(model: svm, x_test: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    y_pred = []
    for i in tqdm(range(0, x_test.shape[0], batch_size)):
        x_batch = x_test[i: i + batch_size]
        y_batch = predict(model, x_batch)
        y_pred.append(y_batch)
    return np.concatenate(y_pred)

def run_inference_process_pool(model: svm, x_test: np.ndarray, max_workers: int = 16) -> np.ndarray:
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        chunk_size = len(x_test) // max_workers

        # Split the test data into chunks and submit each chunk to a worker process
        futures = []
        for i in range(0, len(x_test), chunk_size):
            x_test_chunk = x_test[i:i + chunk_size]
            future = executor.submit(run_inference, model, x_test_chunk)
            futures.append(future)

        # Wait for the worker processes to complete and collect the results
        completed_futures, _ = concurrent.futures.wait(futures)
        y_pred = [future.result() for future in completed_futures]

    return np.concatenate(y_pred)

@ray.remote
def run_inference_chunk(model: svm, x_test_chunk: np.ndarray) -> np.ndarray:
    return run_inference(model, x_test_chunk)

def run_inference_process_pool_ray(model: svm, x_test: np.ndarray, max_workers: int = 16) -> np.ndarray:
    ray.init(num_cpus=max_workers)

    # Split the test data into chunks
    batch_size = len(x_test) // max_workers
    chunks = [x_test[i:i + batch_size] for i in range(0, len(x_test), batch_size)]
    num_chunks = len(chunks)

    # Submit each chunk to a remote worker process
    futures = [run_inference_chunk.remote(model, chunk) for chunk in chunks]

    # Wait for the worker processes to complete and collect the results
    completed_futures, _ = ray.wait(futures, num_returns=num_chunks)
    y_pred = ray.get(completed_futures)

    ray.shutdown()

    return np.concatenate(y_pred)

def run_inference_process_pool_dask(model: svm, x_test: np.ndarray, max_workers: int = 16) -> np.ndarray:
    batch_size = len(x_test) // max_workers
    cluster = dd.LocalCluster(processes=True, threads_per_worker=1, n_workers=max_workers)
    client = dd.Client(cluster)

    # Split the test data into chunks
    chunks = [x_test[i:i + batch_size] for i in range(0, len(x_test), batch_size)]

    # Submit each chunk to a remote worker process
    futures = [dask.delayed(run_inference)(model, chunk) for chunk in chunks]

    # Compute the predictions in parallel using Dask
    y_pred = dask.compute(*futures)

    client.shutdown()
    cluster.close()

    return np.concatenate(y_pred)

def run():
    x_train, y_train, x_test = get_data()
    model = train_model(x_train, y_train)
    ways = ['simple', 'batch', 'pool', 'ray', 'dask']
    for way in ways:
        start_save = time.monotonic()
        if way == 'simple':
            predict(model, x_test)
        if way == 'batch':
            run_inference(model, x_test)
        if way == 'pool':
            run_inference_process_pool(model, x_test)
        if way == 'ray':
            run_inference_process_pool_ray(model, x_test)
        if way == 'dask':
            run_inference_process_pool_dask(model, x_test)

        end_save = time.monotonic()
        process_time = end_save - start_save
        print(f"{way} - process time: {process_time:.6f}")

if __name__ == "__main__":
    run()