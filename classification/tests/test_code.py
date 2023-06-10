import pytest
import numpy as np
from classification.trainer import compute_metrics
import wandb
from transformers import EvalPrediction


@pytest.fixture
def example_predictions_labels():
    # Generate random predictions and labels
    num_examples = 100
    num_classes = 10
    predictions = np.random.rand(num_examples, num_classes)
    labels = np.random.randint(num_classes, size=num_examples)

    _eval_pred = EvalPrediction(label_ids=labels, predictions=predictions)
    return _eval_pred


def test_compute_metrics(example_predictions_labels: EvalPrediction):
    predictions, labels = example_predictions_labels

    correct_predictions = np.argmax(predictions, axis=1) == labels
    expected_acc = np.sum(correct_predictions) / len(labels)

    actual_metrics = compute_metrics(example_predictions_labels)
    actual_acc = actual_metrics['accuracy']

    # Check if actual accuracy and precision match expected values
    assert np.isclose(actual_acc['accuracy'], expected_acc, rtol=1e-3), "Accuracy is incorrect"
