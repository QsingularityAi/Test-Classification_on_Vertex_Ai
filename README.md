# Text Classification Project

This project demonstrates various approaches to text classification using Keras and TensorFlow, including traditional Keras models and advanced techniques like BERT. It includes implementations in Jupyter Notebooks, covering different neural network architectures and methodologies for text classification.

## Project Overview

This repository contains a collection of Jupyter Notebooks that explore text classification using different deep learning models and techniques:

*   `keras_for_text_classification.ipynb`: Provides a hands-on guide to building and training text classification models using Keras, exploring Deep Neural Networks (DNN), Recurrent Neural Networks (RNN), and Convolutional Neural Networks (CNN) for classifying news article titles based on their source.
*   `classify_text_with_bert.ipynb`: Demonstrates text classification using BERT for sentiment analysis of movie reviews. It covers loading pre-trained BERT models from TensorFlow Hub, building a classifier, fine-tuning, saving, and evaluating the BERT model on the IMDB dataset.
*   `fine_tune_bert.ipynb`: Focuses on fine-tuning BERT models using the `tensorflow-models` package for paraphrase detection on the GLUE MRPC dataset. It includes data preprocessing using the official BERT tokenizer and setting up training pipelines with custom optimizers.
*   `rnn_encoder_decoder.ipynb`: Showcases text classification using RNN encoder-decoder models, providing an alternative approach to sequence-based text classification (details to be added based on notebook content).

## Notebook Contents

*   **`keras_for_text_classification.ipynb`**:
    1.  **Data Acquisition and Preprocessing:** Uses BigQuery to create a dataset of news article titles, preprocesses text data (tokenization, integerization), and prepares data for Keras models.
    2.  **Model Building and Training:** Implements and trains Bag-of-Words DNN, RNN (GRU), and CNN models for text classification.
    3.  **Model Evaluation and Comparison:** Evaluates and compares the performance of different Keras models based on accuracy and loss.

*   **`classify_text_with_bert.ipynb`**:
    1.  **BERT Model Loading and Preprocessing:** Loads pre-trained BERT models and corresponding preprocessing models from TensorFlow Hub. Preprocesses text data (IMDB movie reviews) for BERT input.
    2.  **Classifier Building and Fine-tuning:** Builds a classification model on top of the BERT encoder, defines loss function (BinaryCrossentropy) and optimizer (AdamW), and fine-tunes the model.
    3.  **Model Saving and Evaluation:** Saves the fine-tuned BERT model and evaluates its performance on the test set. Demonstrates inference with the saved model.

*   **`fine_tune_bert.ipynb`**:
    1.  **BERT Fine-tuning using TensorFlow Models:** Utilizes the `tensorflow-models` official package for fine-tuning BERT.
    2.  **Data Preprocessing for BERT:** Uses the official BERT tokenizer and preprocessing steps to prepare the GLUE MRPC dataset (paraphrase detection).
    3.  **Model Building and Training:** Builds a BERT classifier using configurations and models from the `tf-models-official` library, sets up the AdamW optimizer with a learning rate schedule, and trains the model. Includes steps for saving the model and re-encoding large datasets using TFRecord.

*   **`rnn_encoder_decoder.ipynb`**:
    1.  **Dataset Creation for Seq2Seq:** Creates `tf.data.Dataset` suitable for sequence-to-sequence tasks (Spanish-to-English translation).
    2.  **RNN Encoder-Decoder Model:** Implements and trains an RNN (GRU) encoder-decoder model using Keras functional API for translation.
    3.  **Separate Encoder/Decoder Saving:** Saves the trained encoder and decoder as separate Keras models.
    4.  **Translation Implementation:** Implements a decoding function using the separate encoder and decoder models to perform translation.
    5.  **BLEU Score Evaluation:** Evaluates the translation quality using the BLEU score metric.

## Learning Objectives

By exploring this project, my goal is to learn how to:

*   Build and train various text classification models using Keras (DNN, RNN, CNN).
*   Utilize pre-trained BERT models from TensorFlow Hub for text classification tasks like sentiment analysis.
*   Fine-tune BERT models using the `tensorflow-models` package for specific NLP tasks like paraphrase detection.
*   Understand and implement the necessary text preprocessing steps for different models (Keras Tokenizer, BERT Tokenizer).
*   Configure and use optimizers like AdamW with learning rate schedules for fine-tuning large models.
*   Build custom classifiers on top of pre-trained encoders like BERT.
*   Save, load, and evaluate trained models in TensorFlow.
*   Work with different text datasets (BigQuery news titles, IMDB reviews, GLUE MRPC, Spanish-English parallel corpus).
*   Implement and understand RNN encoder-decoder models for machine translation.
*   Create `tf.data.Dataset` pipelines for sequence-to-sequence tasks.
*   Evaluate machine translation models using the BLEU score.

## Files

*   `classify_text_with_bert.ipynb`: Jupyter Notebook demonstrating text classification using BERT.
*   `fine_tune_bert.ipynb`: Jupyter Notebook for fine-tuning BERT models for text classification tasks.
*   `keras_for_text_classification.ipynb`: Jupyter Notebook containing the code and explanations for text classification using Keras.
*   `rnn_encoder_decoder.ipynb`: Jupyter Notebook showcasing text classification using RNN encoder-decoder models.
*   `LICENSE`: GPL-2.0 license file specifying the terms of use and distribution for this project.
*   `README.md`: This file, providing an overview of the project.

## License

This project is licensed under the GPL-2.0 license - see the [LICENSE](LICENSE) file for details.
