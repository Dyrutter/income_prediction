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

## [app.py](./App/data.py)
+ Creates a base model class, using pydantic's Field to convert hyphenated column names to underscored column names
+ Configure the base model class with a schema example
+ Define a [GET] method at the root directory
+ Define a [POST] method which predicts the income category of an individual in the /predict_salary subdirectory

## [test_app.py](./App/test_app.py)
+ Tests functions which download, preprocess, and split the data
+ Asserts the [GET] method is operational
+ Tests a prediction of salary >50k
+ Tests a prediction of salary <=50k

## [post_to_render.py](./App/post_to_render.py)
+ Posts a single data sample to Render
+ Recieves an income prediction for the sample
+ Asserts the link is operational based on its status code

## [sanity_check.py](./App/sanity_check.py)
+ Asserts files exist
+ Unit tests functions
+ Checks GET and POST status code
