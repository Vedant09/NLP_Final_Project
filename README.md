## Group Members:

- HARSHA MASANDRAPALYA VANARAJAIAH</hr>
- VEDANT VENKATESH YELSANGIKAR</hr>
- KAVYA TOLETY

## The Original Research Paper Source : https://aclanthology.org/2023.emnlp-main.810/

## Overview :

We have Replicated the original paper "Token Prediction as Implicit Classification to Identify LLM-Generated Text" for the CS678 Final project, class of Spring 2024

## Checkpoint 2, Reproduce :

Clone the project repository from the following link: **TOBDO**
Then, upload it to Google Drive. We used Google Colab to run this project, so you can access the code there.

## Requirement :

1. Run `!pip3 install -r requirements.txt` to install dependencies.

2. To set the PYTHONPATH environment variable in Google Colab:
   import os

## Set PYTHONPATH to include the project directory

project_path = "/content/drive/MyDrive/T5-Sentinel-public"
os.environ['PYTHONPATH'] += f":{project_path}"

## Optional: Print PYTHONPATH to verify the change

print("PYTHONPATH:", os.environ['PYTHONPATH'])

## Training :

The datasets have already been generated and downloaded to the appropriate folders. You can find them here: **TOBDO**

    1.  `To Train the t5_sentienl model` - !python3 detector/t5_sentinel/_main_.py

**NOTE** : When executing the training command, authentication through Wandb is required. Choose option 2 to indicate that you have an account, and then input the following API value - b0a6fecc14383f64f17b5bbd81acc18b7d864c55 to initiate model training.
You can find the Python notebook for reference at the following link: Ckeckpoint2-implementation.ipynb (https://github.com/Vedant09/NLP_Final_Project/blob/main/Ckeckpoint2-implementation.ipynb)
## Observations :

    Trainig accuracy:
    Hyperparameters used:

    Try it out :  To Change the Hyperparameters you can go the location :

    we can observe the metrics output using wihci we have evaluated our model and they are

    The model evaluation metrics which were used to evaluate our model are as follows:
        Accuracy: 0.6555989583333334
        F1 Score: 0.46038681497949446
        Recall: 0.7108254895596667
        Precision: 0.603226719335036
        Confusion Matrix Accuracy:  [[519  30 241   0]
 [ 15 729  46   0]
 [ 44  10 736   0]
 [400  51 211  30]]

Observations on error analysis of sample output:
Text: de la circunscripción pro-armas. Todo el campo, es decir, a excepción de Donald Trump.El magnate inmobiliario y la estrella de televisión de realidad han optado, al menos hasta ahora;Es notoriamente impredecible cuando se trata de sus planes políticos.* Y yo digo: ¡Buen Ridance!Fue bastante malo tenerlo en el último debate. *Por lo que queremos decir que si estás pasando por el infierno, sigue en curso de todos modos.
Predicted Label: 2
True Label: 3
Plotted confusion matrix image can be found here - 
