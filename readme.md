## Machine Learning Algorithm Demos

This repository contains Python demonstrations of various machine learning algorithms. Below are the dependencies required to run the code.

## Dependencies

Ensure you have the following versions installed:

- Python: **3.12**
- pip: **25.0.1**
- scikit-learn: **1.5.2**
- numpy: **2.0.2**
- matplotlib: **3.9.2**
- pandas: **2.2.3**

## Installation

You can install the dependencies using either a Python virtual environment or Anaconda.

### Using Python Virtual Environment

1. Create a virtual environment:
   ```sh
   python -m venv ml_env
   ```
2. Activate the virtual environment:
   - **Windows**:
     ```sh
     ml_env\Scripts\activate
     ```
   - **Mac/Linux**:
     ```sh
     source ml_env/bin/activate
     ```
3. Upgrade pip and install dependencies:
   ```sh
   pip install --upgrade pip
   pip install scikit-learn==1.5.2 numpy==2.0.2 matplotlib==3.9.2 pandas==2.2.3
   pip install joblib
   pip install xgboost #(xgboost-cpu for apple silicon)
   ```

### Using Anaconda

1. Create a new Anaconda environment:
   ```sh
   conda create --name ml_env python=3.12
   ```
2. Activate the environment:
   ```sh
   conda activate ml_env
   ```
3. Install dependencies:
   ```sh
   conda install scikit-learn=1.5.2 numpy=2.0.2 matplotlib=3.9.2 pandas=2.2.3
   ```

## Running on Google Colab

You can also run these machine learning demos in [Google Colab](https://colab.research.google.com/), an online Jupyter notebook environment.

### Steps to Run on Google Colab:

1. Upload the repository to your Google Drive or clone it using Git:
   ```sh
   !git clone https://github.com/NareshModina/ML.git
   ```
2. Navigate to the project directory:
   ```sh
   %cd your_repository
   ```
3. Install the required dependencies:
   ```sh
   !pip install scikit-learn==1.5.2 numpy==2.0.2 matplotlib==3.9.2 pandas==2.2.3
   ```
4. Open and run the `.ipynb` notebooks in Colab.

## Algorithm Capabilities

The table below indicates whether an algorithm can be used for classification or regression:

| Algorithm            | Classification | Regression |
|----------------------|---------------|-----------|
| Linear Regression    | ❌            | ✅        |
| Logistic Regression  | ✅            | ❌        |
| SVMs                | ✅            | ✅        |
| SGD                 | ✅            | ✅        |
| Decision Tree       | ✅            | ✅        |
| Bagging Trees       | ✅            | ✅        |
| Random Forest       | ✅            | ✅        |
| Gradient Boosting   | ✅            | ✅        |

[!NOTE] 
Sub directory basic contains python files for all the algorithms listed in the table for both regression and classification task

Enjoy experimenting with machine learning algorithms!

# Comparison of ML algorithms on different datasets

## Regression comparison
Three datasets namely Fish dataset, insurance dataset, and nycyellowtaxiride dataset are used for the comparison
-  Sub directory --> algo_comparison/regression_comparison_basic contains interactive python notebooks (ipynb) with the comparison of Linear regression, polynomial regression, SVR, decision tree on all three datasets
- Sub directory --> algo_comparison/regression_comparison_trees contains interactive python notebooks with the comparison of Decision tree, Bagging trees, random forest, gradient boosting (even xgboost)
## Classification comparison (in progress)
Datasets: 
- Sub directory --> algo_comparison/classification_comparison_basic contains the ipy notebooks with the comparison of logistic regression, svm, decision trees

- Sub directory --> algo_comparison/classification_comparison_trees contains interactive python notebooks with the comparison of Decision tree, Bagging trees, random forest, gradient boosting (even xgboost)