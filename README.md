# Text Classification with Keras

This project demonstrates text classification using Keras, as implemented in the `keras_for_text_classification.ipynb` notebook. It explores different neural network architectures for classifying news article titles based on their source (Github, TechCrunch, or New York Times).

## Project Overview

This repository contains a Jupyter Notebook (`keras_for_text_classification.ipynb`) that provides a hands-on guide to building and training text classification models using Keras. The notebook covers data preparation, model building, training, and evaluation using Deep Neural Networks (DNN), Recurrent Neural Networks (RNN), and Convolutional Neural Networks (CNN).

## Notebook Contents

The notebook is structured to guide you through the following key steps:

1.  **Data Acquisition and Preprocessing:**
    *   Utilizes BigQuery to create a dataset of news article titles and their sources.
    *   Preprocesses text data by tokenizing and integerizing the titles using Keras Tokenizer.
    *   Demonstrates one-hot encoding of labels for multi-class classification.
    *   Splits the dataset into training and validation sets.

2.  **Model Building and Training:**
    *   **Bag-of-Words DNN Model:** Implements a simple DNN model that averages word embeddings to create a bag-of-words representation, followed by dense layers for classification.
    *   **RNN (GRU) Model:** Builds an RNN model using GRU layers to capture sequential information in the text, considering word order for improved classification.
    *   **CNN Model:** Develops a CNN model using 1D convolutional layers to extract n-gram features from the text, offering another approach to sequence-based classification.

3.  **Model Evaluation and Comparison:**
    *   Trains each model and evaluates its performance using validation accuracy and loss.
    *   Visualizes training and validation loss and accuracy curves for each model.
    *   Compares the performance of DNN, RNN, and CNN models in terms of training speed and maximum validation accuracy.

## Learning Objectives

By exploring this project, you will learn how to:

*   Create text classification datasets using BigQuery.
*   Tokenize and integerize text corpora for Keras models.
*   Perform one-hot encoding of categorical labels in Keras.
*   Utilize embedding layers to represent words as dense vectors.
*   Understand and implement bag-of-words representations.
*   Build and train DNN, RNN, and CNN models for text classification in Keras.
*   Evaluate and compare the performance of different neural network architectures for text classification.

## Files

*   `keras_for_text_classification.ipynb`: Jupyter Notebook containing the code and explanations for text classification using Keras.
*   `LICENSE`: GPL-2.0 license file specifying the terms of use and distribution for this project.
*   `README.md`: This file, providing an overview of the project.

## License

This project is licensed under the GPL-2.0 license - see the [LICENSE](LICENSE) file for details.
