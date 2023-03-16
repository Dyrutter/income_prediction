# PROJECT OVERVIEW
This is a demonstration of how to deploy a machine learning model using fast API. 

The project follows these steps:

+ Using census data, train a classification model to predict and categorize an individual's salary. The threshold salary used is $50,000. A more detailed explanation of the model and data set is available here: [model_card](./model_card)
+ Create unit tests to monitor the model's performance on various slices
+ Deploy the model using the FastAPI package and create API tests
+ Incorporate slice validation and API tests into a CI/CD framework using GitHub Actions
+ Run flake8 and pytest on every GitHub push and pull
+ Upload model to Render for real-time analysis

# ENVIRONMENT SETUP
## PREREQUISITES

+ Render account
+ GitHub account
+ Clone GitHub repo
`https://github.com/Dyrutter/income_prediction.git`

## DEPENDENCIES
+ Install requirements found in [requirements](./requirements.txt)
+ MacOS Monterey was used with python 3.10 inside a `conda` virtual environment

# MAIN FILE DESCRIPTIONS

## [data.py](./App/data.py)
+ Downloads and preprocesses census data
+ Finds Means and Std devs of feature-specific data slices
+ Trains both a logistic regression and random forest model
+ Produces Images:
  + [Classification Reports](./data/figure_file/classification_reports.png)
  + [Roc Plots](./data/figure_file/roc_curves.png)
  + [A Feature Importance Plot](./data/figure_file/feature_importances.png)
  + [A txt File Showing Performance Metrics on Specific Categorical Features](./data/slice_metrics.txt)
