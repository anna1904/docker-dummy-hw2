train_example:
	export WANDB_PROJECT=classification_example
	python3 classification/cli.py training

upload_example:
	python3 classification/cli.py upload-to-registry classification_example classification/my_classification_model/checkpoint-10/ classification_example