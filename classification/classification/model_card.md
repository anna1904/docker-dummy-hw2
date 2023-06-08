---
language: en
name: Image Classification for war losses
model_type: image-classification
dataset_tags:

- images
- losses
- war
  model_tags:
- vit
- transformer
- huggingface

---

# Model Card for losses-image-classification-vit

<!-- Provide a quick summary of what the model is/does. -->

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

The Losses Classification Model is a machine learning model that can analyze images of losses in war and classify them
into different categories. The dataset used to train the model consists of images of russian losses in the
russo-Ukrainian War of 2022. This model can assist war experts in classifying images for analytics.

- **Developed by:** Anna Konovalenko
- **Model type:** image-classification
- **Language(s) Python:** en
- **License:**+
- **Finetuned from model [optional]: facebook/deit-tiny-patch16-224

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **
  Repository:** https://wandb.ai/projector-team/registry/model?selectionPath=projector-team%2Fmodel-registry%2Fmodel-losses&view=membership&tab=overview&version=v0

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

The model can categorise images up to 18 classes.

The classes are the following:

- Aircraft
- Armoured_Fighting_Vehicles
- Armoured_Personnel_Carriers
- Artillery_Support_Vehicles_And_Equipment
- Command_Posts_And_Communications_Stations
- Engineering_Vehicles_And_Equipment
- Helicopters
- Infantry_Fighting_Vehicles
- Infantry_Mobility_Vehicles
- Mine-Resistant_Ambush_Protected
- Multiple_Rocket_Launchers
- Reconnaissance_Unmanned_Aerial_Vehicles
- Self-Propelled_Anti-Tank_Missile_Systems
- Self-Propelled_Artillery
- Surface-To-Air_Missile_Systems
- Tanks
- Towed_Artillery
- Trucks,_Vehicles_and_Jeeps

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model.

## Training Details

### Training Data

<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

https://www.kaggle.com/datasets/piterfm/2022-ukraine-russia-war-equipment-losses-oryx

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

The image preprocessing involves a series of steps to prepare the image for further analysis. The initial step is to
obtain the image processor model and its associated checkpoint. The image is then normalized using the mean and standard
deviation values provided by the image processor. The size of the image is determined by the height and width specified
in the image processor. The image is then transformed through a series of operations including random resizing and
cropping, conversion to a tensor, and final normalization using the previously calculated mean and standard deviation
values.

#### Training Hyperparameters

- **Training
  regime:** 10
  epochs <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

The training dataset consists of 5996 images, where 34 images are for evaluation.
It takes around 1 hour to train the model for these results.

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

Taken from the same repository.

#### Metrics

Two metrics measure the performance of the model: accuracy and precision.

### Results

accuracy: 0.53859
precision: 0.37624
eval_runtime: 15.1723
eval_samples: 596

#### Summary

#### Hardware

MacBook Air (M1, 2020), Memory: 16GB

## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

## Model Card Contact

Anna Konovalenko


