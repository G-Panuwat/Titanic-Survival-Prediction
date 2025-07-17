# Titanic Survival Prediction

This project is a machine learning solution for predicting passenger survival on the Titanic, based on data from the famous Kaggle competition: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic).

## 📚 Overview

The notebook walks through the full machine learning pipeline:

- Data loading and exploration
- Data cleaning and preprocessing
- Feature engineering
- Model training using various algorithms
- Model evaluation
- Submission file generation

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## 📁 Dataset

- `train.csv` and `test.csv` from the Titanic dataset available on Kaggle.
- Make sure to download the dataset from [here](https://www.kaggle.com/competitions/titanic/data) and place it in the same directory as the notebook.

## Methodology

### 1. Download Acquisition
  - Dataset Download: The dataset for this project was obtained from [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic).
### 2. Exploratory Data Analysis (EDA) and Data Cleaning
  - Initial Data Exploration: Conducted simple exploratory data anaylsis to understand the dataset's structure, distribution (mean, median, standard deviation, etc.).
  - Outlier Detection and Handling: Identified and addressed outliers that could impact to model performance, since there are small amount of instances. The data was visualized with scatterplot to determine the impactful outliers.
### 3. Feature Engineering
  - Feature Extraction: Transformed raw features and created new, more informative features to enhance the model prediction. This involved:
    - 'Deck' feature was extracted from the 'Cabin' column, representing the deck level. This was then further categorized or mapped to a 'Deck Class' that correlates with 'Pclass' ranks (1, 2, 3), aiming to capture spatial or social hierarchies.
    - 'Family_Size' feature was created by summing 'SibSp' (number of siblings/spouses aboard) and 'Parch' (number of parents/children aboard) and adding 1 (for the passenger themselves), to represent the total number of family members.
    - 'Family_Size_Grouped' feature was categorized into discrete groups: 'Alone' (for 1 person), 'Small' (for 2-4 persons), 'Medium' (for 5-7 persons), and 'Large' (for 8-11 persons). This grouping helps capture non-linear relationships and potentially reduce noise.
    -  Title Extraction: Titles (e.g., Mr., Mrs., Master, Miss.) were extracted from the 'Name' feature, serving as a proxy for social status and age estimation.
### 4. Feature Preoprocessing
  - Encoding Features:
    - One-Hot Encoding: Applied to nominal categorical features such as 'Sex', 'Embarked', 'Deck', and 'Title'.
    - Ordinal Encoding: Applied to the 'Family_Size_Grouped' feature, as it possesses an inherent ordered relationship (Alone < Small < Medium < Large).
    - Impute NaN values with mean.
### 5. Data Pipeline Construction
  - Automated Workflow: A robust data preprocessing and feature engineering pipeline was built using scikit-learn's 'Pipeline'
### 6. Data Splitting
  - Train-Test Split: The dataset was split into training and testing sets (validation set)
    - The data was split into an 80% training set and a 20% testing set (validation set) using 'train_test_split' with 'random_state'.
### 7. Model Training and Hyperparameter Optimization
  - Model Selection: Explored and trained various machine learning models suitable for the problem.
    - Logistic Regression, Support Vector Machine, Decision Tree, K-nearest neighbors, Naive Bayes, XGB Classifier, and ensemble models (Random Forest, AdaBoost, GradientBoost).
  - Hyperparameter Optimization: Find the best-performing parameters for each model.
    - The given dataset is relatively small, Grid Search Optimization with 5 fold cross validation was chosen for its exhaustive search capability and robustness in finding optimal hyperparameters, mitigating overfitting to the training data.
### 8. Model Evaluation
  - Performance Assessment: The best-performing model was evaluated on the unseen test dataset.
    - The model was evaluated with Accuracy, because the competition goal is to predict passenger survival only.
    - Results: The model achieved an Accuracy of 82.68% on test dataset (from test_train_split), and 79.43% on original Kaggle test dataset.

## 🚀 How to Run

1. Clone the repository or download the `.ipynb` file.
2. Download the Titanic dataset (`train.csv`, `test.csv`) from Kaggle and place them in the working directory.
3. Install the required Python libraries (if not already installed):

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
