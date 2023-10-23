import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import easyocr
import speech_recognition as sr
import audioop
import math
import requests

import tempfile
from nudenet import NudeDetector


# Load the model that you just saved
lr = joblib.load("model.pkl")
cv=joblib.load('vectorizer.pkl')
tfidf=joblib.load('transformer.pkl')


app = Flask(__name__)
CORS(app)
lst=[]
# Define a POST endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the request

    input_text = request.json['text']
    print(input_text)
    if not input_text:
        print("empty")


# Transform the input text using the loaded vectorizer and transformer
    input_text_cv = cv.transform([input_text])
    input_text_tf = tfidf.transform(input_text_cv)

    # Make a prediction on the transformed input text
    prediction = lr.predict(input_text_tf)[0].tolist()

    # Return the prediction as a JSON response
    # prediction=jsonify({'prediction': prediction})
    sentiments = ["age", "not bullying", "miscellaneous","racism", "religion", "gender"]
    all_categories_names = np.array(sentiments)
    print(all_categories_names[prediction])
    return all_categories_names[prediction]

@app.route('/predict', methods=['GET'])
def web_scrape():
    # "C:\Users\sreem\Downloads\chromedriver_win32\chromedriver.exe"
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service

    # Set up Chrome options
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')  # Run Chrome in headless mode

    # Set up ChromeDriver executable path
    chromedriver_path = r'C:\Users\sreem\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe'
    service = Service(executable_path=chromedriver_path)
    # Create a new ChromeDriver instance with the service and options
    driver = webdriver.Chrome(service=service, options=chrome_options)

    driver.get('http://localhost:8080')

    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException
    import requests

    # Wait for the div to be loaded
    try:
        # Wait for the div element to be present
        divs = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.q-list.q-pa-sm'))
        )

        # Print the text contents of the div element
        lst.clear()
        res=[]
        for div in divs:
            tweets = div.text.split('\n')
            lst.append(tweets[1])
        for i in lst:
            url = 'http://127.0.0.1:5000/predict'
            myobj = {'text': i}

            x = requests.post(url, json=myobj)
            res.append(x.text)
        return res





    except TimeoutException:
        print("Timed out waiting for the div element to be present")

    finally:
        # Quit the driver
        driver.quit()


@app.route('/tweets', methods=['GET'])
def tweets():
    return lst


@app.route('/transcribe', methods=['POST'])
def transcribe():
    file = request.files['audio']
    file.save(r"C:\Users\sreem\PycharmProjects\cyberbullying\audio.mp3")




    filename = r"C:\Users\sreem\PycharmProjects\cyberbullying\audio.mp3"
    # initialize the recognizer
    r = sr.Recognizer()
    # open the file
    with sr.AudioFile(filename) as source:
        # listen for the data (load audio to memory)
        audio_data = r.record(source)
        # recognize (convert from speech to text)
        text = r.recognize_google(audio_data)

    with open(filename, 'rb') as audio_file:
        audio_data = audio_file.read()
        rms = audioop.rms(audio_data, 2)  # Calculate root mean square
        sound_intensity_db = 20 * math.log10(rms)
    print(text, sound_intensity_db)
    return jsonify({
    "text": text,
    "loudness": sound_intensity_db
})



# Initialize the EasyOCR reader with the desired language(s)
reader = easyocr.Reader(['en'])

@app.route('/extract_text', methods=['POST'])
def extract_text():
    # Check if an image file is included in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files['image']

    # Read the image data as bytes
    image_bytes = image.read()

    # Perform OCR on the image
    result = reader.readtext(image_bytes)

    # Extract and join the recognized text
    extracted_text = ' '.join([detection[1] for detection in result])

    return jsonify({"text": extracted_text})

@app.route('/detectnudity', methods=['POST'])
def detect_nudity():
    if request.json["url"] == "":
        return "NA"
    image_url = request.json["url"]

    response = requests.get(image_url)

    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(response.content)
            image_path = temp_file.name

        # Detector
        detector = NudeDetector()

        results = detector.detect(image_path)

        is_offensive = any('class' in result and result['class'] in [
            "FEMALE_GENITALIA_COVERED",
            "BUTTOCKS_EXPOSED",
            "FEMALE_BREAST_EXPOSED",
            "FEMALE_BREAST_COVERED",
            "FEMALE_GENITALIA_EXPOSED",
            "MALE_BREAST_EXPOSED",
            "ANUS_EXPOSED",
            "BELLY_EXPOSED",
            "MALE_GENITALIA_EXPOSED",
            "ANUS_COVERED",
            "BUTTOCKS_COVERED", ] for result in results)

        if is_offensive:
            print("Yes")
            return "Yes"
        else:
            return "No"
    else:
       return "Failed to fetch the image from the URL"


# Run the app
if __name__ == '__main__':
    app.run(debug=True)




