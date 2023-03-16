# PROJECT OVERVIEW
This is a demonstration of how to deploy a machine learning model using fast API. 

The project follows these steps:

+ Using census data, train a classification model to predict and categorize an individual's salary. The threshold salary used is $50,000. A more detailed explanation of the model is available here: [model_card](./model_card)
Expose the model for inference using a FastAPI app
Deploy the app using Heroku to provide inference endpoint
Implement Continuous Integration / Continuous Deployment workflow using Github actions, github repository and Heroku integration with Github. The app is only deployed if integrated, automated, tests are validated by Github actions upon modifications
