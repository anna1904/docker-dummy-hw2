from huggingface_hub import ModelCard, ModelCardData


def create_model_card():
    card_data = ModelCardData(
        name="Image Classification for war losses",
        language="en",
        model_type="image-classification",
        dataset_tags=["images", "losses", "war"],
        model_tags=["vit", "transformer", "huggingface"],
    )
    results_data = ModelCardData(
        accuracy=0.53859,
        precision=0.37624,
        eval_runtime=15.1723,
        eval_samples=596
    )

    card = ModelCard.from_template(
        card_data,
        model_id="losses-image-classification-vit",
        model_description="The Losses Classification Model is a machine learning model that can analyze images of losses in war and classify them into different categories. The dataset used to train the model consists of images of russian losses in the russo-Ukrainian War of 2022. This model can assist war experts in classifying images for analytics.",
        training_data="https://www.kaggle.com/datasets/piterfm/2022-ukraine-russia-war-equipment-losses-oryx",
        citation="ViT: Vision Transfomer",
        authors=["Anna Konovalenko"],
        codebase="",
        version="v1.0",
        model_sources='facebook/deit-tiny-patch16-224',
        results=results_data,
        evaluation=['accuracy', 'precision'],
        preprocessing="The image preprocessing involves a series of steps to prepare the image for further analysis. The initial step is to obtain the image processor model and its associated checkpoint. The image is then normalized using the mean and standard deviation values provided by the image processor. The size of the image is determined by the height and width specified in the image processor. The image is then transformed through a series of operations including random resizing and cropping, conversion to a tensor, and final normalization using the previously calculated mean and standard deviation values. "
    )

    card.save("model_card.md")


create_model_card()
