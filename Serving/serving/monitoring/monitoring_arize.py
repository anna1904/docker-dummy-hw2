from arize.pandas.logger import Client, Schema
from arize.utils.types import ModelTypes, Environments

import pandas as pd
import datetime
import numpy as np
import os


def get_prediction_file():
    file_url = "https://storage.googleapis.com/arize-assets/documentation-sample-data/data-ingestion/multiclass-classification-assets/multiclass-sample-data.parquet"
    df = pd.read_parquet(file_url)
    current_time = datetime.datetime.now().timestamp()

    earlier_time = (
            datetime.datetime.now() - datetime.timedelta(days=30)
    ).timestamp()

    optional_prediction_timestamps = np.linspace(
        earlier_time, current_time, num=df.shape[0]
    )

    df["prediction_ts"] = pd.Series(optional_prediction_timestamps.astype(int))
    df[["prediction_ts"]].head()
    return df


arize_client = Client(space_key=os.getenv("SPACE_KEY"), api_key=os.getenv("API_KEY"))

schema = Schema(
    prediction_id_column_name="prediction_id",
    timestamp_column_name="prediction_ts",
    prediction_label_column_name="class_pred",
    feature_column_names=["feature1", "feature2", "feature3", "feature4"],
    actual_label_column_name="actual_class"
)
response = arize_client.log(
    dataframe=get_prediction_file(),
    model_id="multiclass-classification-metrics",
    model_version="1.0",
    model_type=ModelTypes.SCORE_CATEGORICAL,
    environment=Environments.PRODUCTION,
    schema=schema
)

if response.status_code == 200:
    print(f"Successfully logged production dataset to Arize")
else:
    print(
        f"Logging failed with response code {response.status_code}, {response.text}"
    )
