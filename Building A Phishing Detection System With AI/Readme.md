
## Table of Contents
1. [Introduction](#introduction)
2. [Methodology](#methodology)
   - [Step 1: Import Dependencies](#step-1-import-dependencies)
   - [Step 2: Load the Dataset](#step-2-load-the-dataset)
   - [Step 3: Dataset Analysis](#step-3-dataset-analysis)
   - [Step 4: Splitting Data](#step-4-splitting-data)
   - [Step 5: Training Decision Tree Algorithm](#step-5-training-decision-tree-algorithm)
   - [Step 6: Model Evaluation Metrics](#step-6-model-evaluation-metrics)
3. [Results](#results)
4. [Conclusion](#conclusion)

## Project Report: Building a Phishing Detection System With AI Using Decision Tree

### Introduction<a name="introduction"></a>
The objective of this project is to develop an AI-based system capable of detecting phishing websites. Phishing is a common cyber threat where attackers attempt to deceive users into revealing sensitive information. Using machine learning, specifically the Decision Tree algorithm, we aim to create a system that can automatically identify and classify phishing URLs.

### Methodology<a name="methodology"></a>

#### Step 1: Import Dependencies<a name="step-1-import-dependencies"></a>
In this step, we imported essential libraries such as NumPy, Pandas, Matplotlib, and scikit-learn. These libraries are crucial for data handling, visualization, and implementing machine learning algorithms.

#### Step 2: Load the Dataset<a name="step-2-load-the-dataset"></a>
We utilized the Phishing Dataset for Machine Learning from Kaggle, which contains features related to URLs and indicators of phishing behavior. The link used is [here](https://www.kaggle.com/datasets/shashwatwork/phishing-dataset-for-machine-learning).

#### Step 3: Dataset Analysis<a name="step-3-dataset-analysis"></a>
Before training our model, we performed exploratory data analysis (EDA) to understand the structure of the dataset, distribution of features, and potential relationships between variables.

#### Step 4: Splitting Data<a name="step-4-splitting-data"></a>
The dataset was divided into two parts: a training set and a testing set. This separation allows us to train the Decision Tree model on one portion of the data and evaluate its performance on unseen data.

#### Step 5: Training Decision Tree Algorithm<a name="step-5-training-decision-tree-algorithm"></a>
We trained a Decision Tree classifier using the training data. This involved teaching the model to recognize patterns and distinguish between phishing and legitimate URLs based on the features provided in the dataset.

#### Step 6: Model Evaluation Metrics<a name="step-6-model-evaluation-metrics"></a>
To assess the effectiveness of our model, we used several metrics such as accuracy, precision, and recall on the test data. These metrics help measure how well the model performs in identifying phishing URLs without misclassifying legitimate ones.

### Results<a name="results"></a>
The Decision Tree model achieved the following performance metrics on the test data:
- Accuracy: 98.44%
- Precision: 98.43%
- Recall: 98.51%

### Conclusion<a name="conclusion"></a>
In conclusion, our phishing detection system based on the Decision Tree algorithm demonstrated high accuracy, precision, and recall rates. This indicates that the model is effective in distinguishing between phishing and non-phishing URLs. Further enhancements and real-time integration could make the system even more robust in combating phishing attacks.
