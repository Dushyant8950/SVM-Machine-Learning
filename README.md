# SVM-Machine-Learning
This code demonstrates the process of building, evaluating, and tuning Support Vector Machine (SVM) classifiers using Python's scikit-learn library on the Iris dataset. Here's a breakdown of each step:

**1. Import Necessary Libraries**

The code begins by importing essential libraries:

- `numpy` and `pandas` for data manipulation.
- Modules from `sklearn` for dataset loading, model building, evaluation, and preprocessing.
- `matplotlib.pyplot` and `seaborn` for data visualization.

**2. Load the Dataset**

The Iris dataset is loaded from `sklearn.datasets`. This dataset contains 150 samples of iris flowers, each described by four features: sepal length, sepal width, petal length, and petal width. The target variable indicates the species, which can be Setosa, Versicolor, or Virginica. The data is then converted into a pandas DataFrame for better visualization and manipulation.

**3. Data Preprocessing**

- **Train-Test Split:** The dataset is split into training and testing sets using a 70-30 ratio to evaluate the model's performance on unseen data.
- **Feature Scaling:** Standardization is applied to the features using `StandardScaler` to ensure that each feature contributes equally to the model's performance.

**4. Building the SVM Model**

An SVM model with a linear kernel is created and trained on the standardized training data. The linear kernel is suitable for linearly separable data.

**5. Model Evaluation**

- **Predictions:** The trained model predicts the species of the test set samples.
- **Accuracy Score:** The proportion of correctly predicted samples is calculated.
- **Confusion Matrix:** A matrix is generated to visualize the performance of the classification model, showing the counts of true positive, true negative, false positive, and false negative predictions.
- **Classification Report:** A detailed report is provided, including precision, recall, and F1-score for each class.
- **Visualization:** A heatmap of the confusion matrix is plotted using Seaborn for better interpretability.

**6. Experiment with Different Kernels**

The code experiments with an SVM model using the Radial Basis Function (RBF) kernel, which is effective for non-linear data. The model is trained, evaluated, and its performance metrics are displayed similarly to the linear kernel model.

**7. Hyperparameter Tuning**

To optimize the SVM model's performance, hyperparameter tuning is performed using `GridSearchCV`. A parameter grid is defined with various values for `C` (regularization parameter), `gamma` (kernel coefficient), and `kernel` type. The grid search performs cross-validation to find the best combination of parameters. The best parameters are then used to make predictions, and the model's performance is evaluated and visualized.

This comprehensive approach ensures that the SVM model is well-tuned and its performance is thoroughly evaluated on the Iris dataset. 
