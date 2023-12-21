# Heart Disease Prediction

The provided Python code is a simple implementation of a logistic regression model for predicting heart disease based on a dataset. Here's a description of the code:

1. **Importing Libraries:**
   - `numpy` (imported as `np`): A library for numerical operations.
   - `pandas` (imported as `pd`): Used for data manipulation and analysis.
   - `matplotlib.pyplot` (imported as `plt`): A plotting library for creating visualizations.
   - `seaborn` (imported as `sns`): Built on top of Matplotlib, it provides an interface for drawing attractive statistical graphics.
   - `train_test_split` from `sklearn.model_selection`: Used to split the dataset into training and testing sets.
   - `LogisticRegression` from `sklearn.linear_model`: Implements logistic regression, a classification algorithm.
   - `accuracy_score` from `sklearn.metrics`: Calculates the accuracy of the model.

2. **Loading and Exploring the Dataset:**
   - The code loads a dataset named `heart_disease_data.csv` using pandas and assigns it to the variable `heart_dataset`.
   - The first few rows of the dataset are displayed using `head()` to provide an overview.
   - Descriptive statistics of the dataset are displayed using `describe()`.

3. **Correlation Heatmap:**
   - The code calculates the correlation matrix of the dataset (`correlation`).
   - It then creates a heatmap using Seaborn to visualize the correlation between different features.

4. **Data Preparation:**
   - The features (`X`) are created by dropping the 'target' column from the dataset.
   - The target variable (`Y`) is assigned the 'target' column.

5. **Train-Test Split:**
   - The dataset is split into training and testing sets using `train_test_split`. The split is 80% training and 20% testing, with stratification based on the target variable for balanced distribution.

6. **Logistic Regression Model:**
   - A logistic regression model is instantiated using `LogisticRegression`.
   - The model is trained on the training data using `fit()`.

7. **Model Evaluation on Training Set:**
   - Predictions are made on the training set (`X_train`) using the trained model.
   - The accuracy of the model on the training set is calculated and displayed.

8. **Model Evaluation on Testing Set:**
   - Predictions are made on the testing set (`X_test`).
   - The accuracy of the model on the testing set is calculated and displayed.

9. **Making Predictions on New Input:**
   - A new input tuple (`new_input`) is created, representing a set of features.
   - The trained model is used to predict the target variable for the new input, and the prediction is displayed.

The code essentially performs data loading, exploration, visualization, model training, and evaluation, and concludes with making predictions on new input data using logistic regression for heart disease classification.
