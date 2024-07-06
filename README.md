<<<<<<< HEAD
# Safe Sonic

This repository contains the code for a Flask-based web application that performs speaker identification using Gaussian Mixture Models (GMM) and Firebase for storage and authentication.

## Features

- **Voice Recording:** Record voice samples for training and testing.
- **Voice Training:** Train GMM models using recorded voice samples.
- **Voice Testing:** Identify the speaker from a test recording.
- **User Signup:** Signup functionality with Firebase Firestore.
- **Model Management:** Rename and delete trained models.
- **Dashboard:** View and manage trained models.

## Prerequisites

- Python 3.6+
- Flask
- Flask-SocketIO
- PyAudio
- Numpy
- Scipy
- scikit-learn
- python_speech_features
- Pydub
- Firebase Admin SDK

## Installation

1. **Clone the repository:**

```sh
git clone https://github.com/rockstar-anjan7/SafeSonic.git
cd speaker-identification-system
```

2. **Install the dependencies:**

```sh
pip install -r requirements.txt
```

3. **Set up Firebase:**

- Create a Firebase project.
- Download the service account key JSON file and save it as `credentials.json`.
- Update `storage.bucket` with your Firebase Storage bucket name.

## Usage

1. **Ensure the project is located in the `D:` drive of your laptop or PC.**

2. **Start the Flask server:**

```sh
python app.py
```

3. **Access the application:**

   Open your web browser and navigate to `http://localhost:5000`.

## Routes

- **GET /**: Home page.
- **GET, POST /record_train**: Record voice samples for training.
- **POST /train**: Train the GMM models with the recorded samples.
- **GET, POST /record_test**: Record voice samples for testing.
- **POST /test**: Identify the speaker from the test recording.
- **GET /get_models**: Retrieve a list of trained models.
- **GET /dashboard**: Dashboard to view and manage trained models.
- **POST /rename_model**: Rename a trained model.
- **POST /delete_model**: Delete a trained model.
- **POST /signup**: User signup.

## Firebase Integration

- **Firestore**: Store user details.
- **Storage**: Save and retrieve voice recordings.

## Additional Functions

- **Feature Extraction**: Extract MFCC features from audio.
- **Delta Calculation**: Calculate delta coefficients for MFCC features.
- **Model Training**: Train GMM models with voice features.
- **Model Testing**: Test the recorded voice against trained models.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Thanks to the developers of the libraries and tools used in this project.

## Important Points

- Change all paths in the code as per your directory.
- Four options are there when you run `app.py`:
  1. Record audio for training
  2. Train Model
  3. Record Audio for Testing
  4. Test Model
  (Follow the same order).
- Store your audio recorded for training in the `training_set` folder and testing audio in the `testing_set` folder.
- Use `training_set_addition.txt` to append trained files and `testing_set_addition.txt` for appending test files.
=======
# Safe_Sonic
Speaker Identification system using machine learning
>>>>>>> dc53531e9ec6d71d62457fccd413b1b1a9363027
