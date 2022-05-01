## Natural Language Inference

This repository consists of a set of experiments on the problem of Natural Language Inference.
These experiments range from statistical models to deep networks that solve the problem efficiently.

The list of experiments/models:
- BoW model
- LSTM based recurrent network for NLI
- DIIN model for NLI

The dataset used in these model is the SNLI dataset released by Stanford.
The link for the same is: https://nlp.stanford.edu/projects/snli/snli_1.0.zip

### Code Structure:
```
project
│   README.md
│   
│
└───data
│   │   snli_1.0_dev.txt
│   │   snli_1.0_test.txt
|   |   snli_1.0_train.txt ## download from site
|   
└───results ## Contains the classification report  
│   └───bertcased
│       │   accuracy.txt
│       │   results.txt
│   └───bertuncased
|   └───distilcased
|   └───distiluncased
|   └───lstm
|   └───bow
|
|───src ## Contains the code for the models
|    |  Cased_DIIN.ipynb ## Contains code base for DIIN model with BERT-Cased Encoder
|    |  Uncased_DIIN.ipynb ## Contains code base for DIIN model with BERT-Uncased Encoder
|    |  distil_Cased_DIIN.ipynb ## Contains code base for DIIN model with DistilBERT-Cased Encoder
|    |  distil_Uncased_DIIN.ipynb ## Contains code base for DIIN model with DistilBERT-Uncased Encoder
|    |  lstm_snli.ipynb ## Contains code base for LSTM model for SNLI classificatio 
 
```

Saved Weights for model: https://drive.google.com/file/d/1pYTNUXvEQjzfv2JcgF45uYBBhFYBJmY4/view?usp=sharing

### Instructions
- Python Version: >3.7

- Modules necessary:
    - PyTorch
    - Transformers
    - PyTorch Lightning (lstm_snli.ipynb)
    - scikit-learn
    - pandas
    - numpy

- Look at the directory structure for files corresponding to each modele

- To run the pretrained model follow these steps:
    - Download the pretrained weights from the drive above
    - Replace the test dataset path with the dataset of choice in Uncased.ipynb

- To train the model on a custom dataset:
    - Place the custom dataset in the `data/` folder
    - Replace the train dataset path with the custom dataset path

### Input Format
All Input needs to be in the same format as `snli` dataset. Look at the links above for the format