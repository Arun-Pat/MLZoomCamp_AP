
## Backgroud
What is Autism
Autism, or autism spectrum disorder (ASD), refers to a broad range of conditions characterized by challenges with social skills, repetitive behaviors, speech and nonverbal communication.

## Problem Statement

This dataset is composed of survey results for more than 700 people who filled an app form. There are labels portraying whether the person received a diagnosis of autism, allowing machine learning models to predict the likelihood of having autism, therefore allowing healthcare professionals prioritize their resources. 

**The Role of Machine Learning**
- Predict the likelihood of a person having autism using survey and demographic variables.
- Explore Autism across Gender, Age, and other variables

**About Dataset**

The data set is taken from kaggle https://www.kaggle.com/datasets/andrewmvd/autism-screening-on-adults/data
Copy of dataset at below location https://raw.githubusercontent.com/Arun-Pat/MLZoomCamp_AP/640c35b994c4f2055f637336455781bbed6ba3b8/MidTerm_Project/autism_screening.csv

This dataset contains 704 rows and 20 columns. The age range is between 17 to 65. Mostly they, 're from USA and UK. The number of males and females is almost equal. In total 26% of candidates are ASD-positive. Jaundice and Family member with ASD are other features which are checked. There are too few nulls, also we have outliers.

Below is description of dataset columns

|Feature |	Description |
|:----------------|:------------------------------------------------|
AX_Score |	Score based on the Autism Spectrum Quotient (AQ) 10 item screening tool AQ-10
age	 | Age in years
gender |	Male or Female
ethnicity |	Ethnicities in text form
jaundice	|Whether or not the participant was born with jaundice?
autism |	Whether or not anyone in tbe immediate family has been diagnosed with autism?
country_of_res |	Countries in text format
used_app_before	| Whether the participant has used a screening app
result |	Score from the AQ-10 screening tool
age_desc |	Age as categorical
relation	 | Relation of person who completed the test
Class/ASD	| Participant classification


## PROJECT FILES
* `README.md` This file
  * Description of the problem
  * Instructions on how to run the project
* Data - `autism_screening.csv`
  * This file contains dataset. Also this dataset availble on kaggle (link in "About dataset" section)
* `notebook.ipynb` with entire code for all tried models   
  * Data analysis ,preparation and data cleansing
  * EDA, feature importance analysis
  * Model selection process and parameter tuning
  * Code in this file is used for creating train.py and predict.py files
* Script `train.py` 
  * Training the best final model
  * Saving it to a file using pickle
* Model `model_A.bin` 
  * Binary pickle file containing Trained ML model and data vector created by running train.py and using training data
* Script `predict_orig.py`
  * Can use this script to test the serialized model and dv, by loading it and checking the result against test input
* Script `predict.py`
  * Loading the model
  * Serving it via a web service (with Flask)
* Script `test_predict.py`
  * Testing the webservice using test data
  * Two samples one which results in "asd" and other No "asd" as out put
* Script `render_test_predict.py`
  * This is modified with host url to testing deployed service on render.com
  * Two samples one which results in "asd" and other No "asd" as out put
* `Pipfile` and `Pipfile.lock`
  * Pipenv file to install required python packages
* `waitress_Pipfile` and `waitress_Pipfile.lock`
  * Pipenv file to install required python packages on windows platform
* `Dockerfile` for running the service
  * File to use with Docker to build the image which then run it in container
* Deployment
  * The service (Docker) has been deployed on cloud `render.com`
    * url `https://predict-cs4g.onrender.com/predict`
  * To interact with the deployed service run render_test_predict.py program 
    * python render_test_predict.py
  * Please wait for response as it is hosted on free plan and becomes inactive after some time and auto-restarts on getting request  
* `AQ10.pdf`
  * A quick referral guide for adults with suspected autism who do not have a learning disability.

    


## INSTRUCTION TO RUN THE PROJECT
- This project is a classification project hence various models below were tried to find best model
	- Logistic regression 
	- Decision Tree Classification
	- Random Forest
	- XG Boost
- Although Logistic regression results were good , The best model seems to be XG Boost which is finally used in creating model
- Runnning train.py will create 'Model_A.bin' which is a serialized model and data vector
- predict.py can be run standalone as a Flask based webservice and test it running test_predict.py
- This can as well be run using waitress or gunicorn server as below and test it running test_predict.py 
  - For windows use waitress > waitress-serve --listen=0.0.0.0:9696 predict:app
  - For Linux/Mac use gunicorn with command as > gunicorn --bind=0.0.0.0:9696 predict:app 
- Command to use install virutal enviornment with  `Pipfile` and `Pipfile.lock` in the same directory. This installs gunicorn WSG Server
    - command to install -> pipenv install
- If you want to install  waitress WSGI server instead of gunicorn  (for windows o/s)
    - rename  `waitress_Pipfile` and `waitress_Pipfile.lock` to  `Pipfile` and `Pipfile.lock` repectively
    - command to install -> pipenv install
- Ensure Dockerfile is there in current directory
    - To build Docker image use command -> docker build -t asd_predict
    - To Run the docker container with image use command -> docker run -it --rm -p 9696:9696 asd_predict
- To test the response for test data run test_predict.py on another terminal -> python test_predict.py 
- The service (Docker) has been hosted on cloud render.com. To test this service use
  - The service has been deployed on cloud `render.com`
    - url `https://predict-cs4g.onrender.com/predict`
  - To interact with the deployed service run render_test_predict.py program 
    - python render_test_predict.py
  - Please wait for response as it is hosted on free plan and becomes inactive after some time and auto-restarts on getting request 

![Output after Running `render_test_predict.py` ](image-1.png)
