

# Building An Email Filtering System With AI

## Table of Contents

1. [Problem Explanation](#problem-explanation)
2. [Importing Libraries](#importing-libraries)
3. [Dataset Exploration](#dataset-exploration)
4. [Data Analysis](#data-analysis)
   - [Distribution of Spam and Ham Messages](#distribution-of-spam-and-ham-messages)
   - [Text Length Analysis](#text-length-analysis)
5. [Selected ML Algorithm Tutorial: K-Nearest Neighbors (KNN)](#selected-ml-algorithm-tutorial-k-nearest-neighbors-knn)
6. [Training and Testing the System](#training-and-testing-the-system)
7. [Creating an OpenAI ChatGPT Version of the System](#creating-an-openai-chatgpt-version-of-the-system)
8. [Comparing Two Systems Together](#comparing-two-systems-together)
9. [Conclusion](#conclusion)


## Problem Explanation

The problem at hand involves classifying messages as either "spam" or "ham" (non-spam). This is a classic example of a binary classification task, where we aim to train a machine learning model that can automatically sort incoming messages based on their content. The significance of solving this problem lies in various applications, such as email filtering systems, SMS spam detection, and maintaining the integrity of messaging platforms.

Spam messages are often unsolicited and can contain phishing links, scams, or unwanted advertisements. Being able to filter these out can save users time and protect them from potential threats. On the other hand, ham messages are regular, non-spam messages that are important to the user. It's crucial that the system has a high accuracy in classifying spam to avoid false positives, which could result in important messages being mistakenly labeled as spam.

### Pseudocode for a Classification Decision

```python
if message_contains_spam_keywords(message):
    classify_as_spam()
else:
    classify_as_ham()
```

The pseudocode above represents a very simplified logic that a spam filter might use, where a message is classified as spam if it contains certain keywords. This is not how modern classifiers work, but it gives a beginner an idea of the decision-making process involved in classification. The actual implementation will use a machine learning algorithm to learn from data which messages are spam or ham.

## Importing Libraries

To work with our dataset and eventually apply the KNN algorithm, we need to import several libraries:

- numpy: Provides support for efficient numerical operations.
- pandas: Essential for data manipulation and analysis.
- matplotlib and seaborn: Used for data visualization.
- sklearn: This is the library that contains a variety of machine learning algorithms, including KNN, as well as utilities for data preprocessing, model evaluation, etc.

## Dataset Exploration

Link to dataset: [Spam Mails Dataset](https://www.kaggle.com/datasets/venky73/spam-mails-dataset/data)

The dataset we are working with is structured as a CSV file with 5171 entries, each representing an email message. It contains the following columns:
- unnamed: an index or identifier for each message.
- label: This is a string indicating whether the message is 'spam' or 'ham'.
- text: The content of the email message.
- label_num: A numerical representation of the label column, where '0' corresponds to 'ham' and '1' corresponds to 'spam'.

## Data Analysis

In this section, we'll take a closer look at our dataset to understand the distribution of classes (spam vs. ham), identify any patterns or anomalies, and prepare the data for the machine learning model. This will involve statistical analysis, visualization, and preprocessing.

### Distribution of Spam and Ham Messages

![Distribution Plot](https://github.com/ijlal321/Integrating-Cyber-Security-With-AI/assets/103317626/4e691ba5-b85d-4e19-9df7-f01d45c8c1d8)

The bar plot above illustrates the distribution of spam and ham messages within our dataset. From this visualization, we can observe whether there's a significant imbalance between the two classes.

Based on the plot, it looks like there are more ham messages than spam messages, which is typical in real-world scenarios where legitimate messages usually outnumber spam. It's important to consider this imbalance when training our machine learning model, as it may lead to a model that's biased towards predicting the majority class.

### Text Length Analysis

The descriptive statistics for the length of messages in each class show that spam messages tend to be slightly longer on average compared to ham messages. However, both types of messages have a wide range of lengths, as indicated by the standard deviation and the maximum length.

Ham messages have a mean length of 977 characters and a maximum length of 32,258 characters. Spam messages have a mean length of 1223 characters and a maximum length of 22,073 characters.

## Selected ML Algorithm Tutorial: K-Nearest Neighbors (KNN)

The K-Nearest Neighbors (KNN) algorithm is a simple, yet effective machine learning algorithm used for classification and regression tasks. In the context of our spam detection problem, we will be using it for classification. KNN works on the principle of feature similarity: a new instance is classified by a majority vote of its neighbors, with the instance being assigned to the class most common among its k nearest neighbors.

For KNN to work with text data, we first need to convert the text into a set of numerical features. This is typically done using techniques like Bag of Words or TF-IDF. We will use the Bag of Words model, which involves tokenization, vocabulary building, and encoding.

## Training and Testing the System

To train and test our KNN model, we'll follow these steps:

1. Split the Data: Divide the dataset into a training set and a testing set.
2. Initialize the KNN Classifier: Choose a value for k and initialize the classifier.
3. Train the Classifier: Fit the classifier to the training data.
4. Test the Classifier: Use the trained classifier to predict the labels of the testing data.
5. Evaluate Performance: Compare the predicted labels to the true labels of the testing set to evaluate the model.

## Creating an OpenAI ChatGPT Version of the System

To create a version of our spam detection system using OpenAI's ChatGPT, we would utilize the OpenAI API to send messages to the model and receive predictions on whether a message is spam or ham. This approach would involve setting up an API call that passes the message text to ChatGPT, which has been fine-tuned on a diverse range of internet text and can perform tasks like text classification when prompted correctly.

## Comparing Two Systems Together

When comparing the performance of our KNN model to that of an OpenAI ChatGPT-based system, we would consider several factors:

- Accuracy: How often does each system correctly identify spam and ham? We can compare the accuracy scores directly if we have equivalent testing data for both systems.
- Scalability: KNN can be computationally expensive as it needs to compute the distance between the input and each training sample. ChatGPT's response time would depend on network latency and server load.
- Cost: Running predictions with KNN is free once the model is trained, but using OpenAI's API can incur costs depending on the number of API calls and the computational resources used.
- Maintainability: Updating the KNN model with new data requires retraining, whereas ChatGPT can potentially adapt to new examples with more sophisticated fine-tuning approaches, assuming OpenAI provides such an interface.
- User Experience: If the system is part of a product, the latency from an API call to OpenAI might affect user experience compared to a locally-run KNN model.

## Conclusion

In conclusion, building an email filtering system with AI involves various steps such as data exploration, analysis, selecting an appropriate machine learning algorithm, training and testing the model, and potentially integrating AI models like ChatGPT for enhanced capabilities.



