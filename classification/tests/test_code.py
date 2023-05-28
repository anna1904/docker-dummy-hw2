import pytest
import numpy as np
from classification.trainer import compute_metrics
import wandb

@pytest.fixture
def example_predictions_labels():
    # Generate random predictions and labels
    num_examples = 100
    num_classes = 10
    predictions = np.random.rand(num_examples, num_classes)
    labels = np.random.randint(num_classes, size=num_examples)
    return predictions, labels

def test_compute_metrics(example_predictions_labels):
    predictions, labels = example_predictions_labels

    wandb.init(
        project="my_project",
        name="test_compute_metrics",
    )

    # Compute expected accuracy and precision
    correct_predictions = np.argmax(predictions, axis=1) == labels
    expected_acc = np.sum(correct_predictions) / len(labels)

    # Compute actual accuracy and precision
    actual_metrics = compute_metrics((predictions, labels))
    actual_acc = actual_metrics['accuracy']

    # Extract numeric values from the returned dictionary
    expected_acc = float(expected_acc)
    actual_acc = float(actual_acc)

    # Check if actual accuracy and precision match expected values
    assert np.isclose(actual_acc, expected_acc, rtol=1e-3), "Accuracy is incorrect"

    # Finish logging
    wandb.finish()

