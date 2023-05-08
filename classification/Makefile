train_example:
	export WANDB_PROJECT=classification_example
	python3 classification/cli.py training

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