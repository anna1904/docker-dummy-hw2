import pytest
import requests
import json


@pytest.fixture(scope="module")
def base_url():
    return "http://0.0.0.0:9000/predict"


def test_inference(base_url):
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {
        "data": {
            "ndarray": [[0, 0], [0, 0]]
        }
    }
    json_payload = json.dumps(payload)

    # Send the POST request
    response = requests.post(base_url, headers=headers, data=json_payload)
    assert response.status_code == 200
    expected_result = 5
    assert len(response.json()['data']["ndarray"]) == expected_result
