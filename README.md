# PROJECT OVERVIEW

This is a demonstration of how to deploy a machine learning model using fast API. 

The project follows these steps:

+ Using census data, train a classification model to predict and categorize an individual's salary. The threshold salary used is $50,000. A more detailed explanation of the model and data set is available here: [model_card](./model_card)
+ Create unit tests to monitor the model's performance on various slices
+ Deploy the model using the FastAPI package and create API tests
+ Incorporate slice validation and API tests into a CI/CD framework using GitHub Actions
+ Run flake8 and pytest on every GitHub push and pull (See: [Continuous Integration](./data/figure_file/continuous_integration.png))
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

# EXAMPLE OUTPUT IMAGES
## A Live Render POST Using CLI Command Prompt

<img width="701" alt="live_script_post" src="https://user-images.githubusercontent.com/126340978/225682482-5df9cef8-476a-4830-bc77-35492bf11a2b.png">

## Live Render GET Using CLI Command Prompt

<img width="446" alt="render_get" src="https://user-images.githubusercontent.com/126340978/225683240-82aecc81-3f90-4b63-ba91-7c1c7094ba5c.png">

## FastAPI POST Localhost Example

<img width="1131" alt="Example" src="https://user-images.githubusercontent.com/126340978/225683975-4e18a1de-4cd1-421e-9f04-482abad4137d.png">

## Live Render Web Service Server

<img width="787" alt="continuous_deployment2" src="https://user-images.githubusercontent.com/126340978/225684597-844c2f9e-8345-437e-a442-00ffddbfb106.png">

## Live Render Webpage Example

<img width="1397" alt="live_post" src="https://user-images.githubusercontent.com/126340978/225687088-79539dff-bfb9-42d8-b651-2a15f11ef003.png">

# ADDITIONAL RESOURCES
+ [ml-ops.org](https://ml-ops.org/)
+ [5 Big Differences Separating API Testing From Unit Testing](https://methodpoet.com/api-testing-vs-unit-testing/)
+ [Why is it Important to Monitor Machine Learning Models?](https://mlinproduction.com/why-is-it-important-to-monitor-machine-learning-models/)

# SUGGESTIONS
+ Use codecov or equivalent to measure the test coverage (then write more tests and try to hit 100% coverage!).
+ Implement FastAPI’s advanced features such as authentication.
+ Create a front end for the API.
