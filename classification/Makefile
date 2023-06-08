train_example:
	export WANDB_PROJECT=classification_example
	python3 classification/cli.py load-mnist-data ./data
	python3 classification/cli.py training ./conf/config.json

upload_model:
	python3 classification/cli.py upload-to-registry classification_example my_classification_model classification_example

test:
	pytest --disable-warnings ./tests/

test_code:
	pytest --disable-warnings -rc ./tests/test_code.py

test_data:
    python3 classification/cli.py load-mnist-data ./data
	pytest --disable-warnings -rc ./tests/test_data.py

test_model:
    python3 classification/cli.py load-mnist-data ./data
	pytest --disable-warnings -ss ./tests/test_model.py

load_data:
	python3 classification/cli.py load-mnist-data ./data

data_inference:
	python3 classification/cli.py run-inference-on-ds ./data/test.ds my_classification_model/ ./tmp/result
