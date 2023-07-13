# OpenAICloudFunction
This project contains the Google Cloud Function used in our prototype.
**Important: You can also run the cloud function locally. However, this does not work on MacOSX.**
Instead, you can start the main.py file without calling the cloud function. We would recommend to run the script, because it is simpler than the cloud function.

## Steps to run the cloud function locally:
1. In your terminal run: ```python3 -m venv env``` 
2. In your terminal run: ```source env/bin/activate```
3. In your terminal run: ```python3 -m pip install -r requirements.txt```
4. Copy the .env.example file and rename it to .env: ```cp .env.example .env```
5. Enter your openai api key in the .env file
6. Run the createDatabase.py file to create the database: ```python3 createDatabase.py```
7. Run the main.py file: ```python3 main.py```
8. If you want to run the cloud function instead, you can use the following command: ```functions-framework --target=http_handler```

## Further information
The cloud function is triggered by a HTTP request. The request must contain the following parameters:
- **question** (string): The question you want to ask the AI

The cloud function returns a JSON object with the following parameters:
- **status** (string): The status code of the request
- **result** (dict): The result of the request
  - *answer* (string): The answer of the AI
  - *sources* (string): The sources used by the AI to answer the question
- **error** (string): The error message if the request failed

If you need further help, you can contact us at: chrissy.drx@gmail.com
