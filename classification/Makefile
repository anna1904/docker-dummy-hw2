train_example:
	export WANDB_PROJECT=classification_example
	python3 classification/cli.py load-mnist-data ./data
	python3 classification/cli.py training ./conf/config.json

upload_example:
	python3 classification/cli.py upload-to-registry classification_example classification/my_classification_model/checkpoint-10/ classification_example

test:
	pytest --disable-warnings ./tests/

test_code:
	pytest --disable-warnings -rc ./tests/test_code.py

test_data:
	pytest --disable-warnings -rc ./tests/test_data.py

test_model:
	pytest --disable-warnings -ss ./tests/test_model.py

load_data:
	python3 classification/cli.py load-mnist-data ./data

data_inference:
	python3 classification/cli.py run-inference-on-dataframe ./data/test.csv classification/my_classification_model/checkpoint-10/ ./data/result.csv
