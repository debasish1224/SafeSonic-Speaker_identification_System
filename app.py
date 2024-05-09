from flask import Flask, render_template, request
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin import storage
from flask_socketio import SocketIO, emit
from flask import jsonify
import os
import wave
import pyaudio
import numpy as np
import time
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
import pickle
import io
from pydub import AudioSegment

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize Firebase Admin SDK
cred = credentials.Certificate('credentials.json')  # Update with your Firebase credentials path
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize Firebase Storage
storage_client = storage.bucket('speaker-identification-system.appspot.com')

@socketio.on('message')
def handle_message(message):
    emit('message', message, broadcast=True)


# Suppressing warnings to avoid cluttering the output
import warnings
warnings.filterwarnings("ignore")

# Path configurations
SOURCE_FOLDER = "training_set"
TEST_FOLDER = "testing_set"
MODEL_FOLDER = "trained_models"
TRAIN_FILE = "training_set_addition.txt"
TEST_FILE = "testing_set_addition.txt"

# Function to calculate delta coefficients for MFCC features
def calculate_delta(array):
    rows, cols = array.shape  # Getting the shape of the array
    deltas = np.zeros((rows, 20))  # Creating an array to store delta coefficients
    N = 2  # Number of frames for delta calculation
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            # Handling edge cases for index calculation
            if i - j < 0:
                first = 0
            else:
                first = i - j
            if i + j > rows - 1:
                second = rows - 1
            else:
                second = i + j
            index.append((second, first))
            j += 1
        # Calculating delta coefficients using neighboring frames
        deltas[i] = (array[index[0][0]] - array[index[0][1]] + (2 * (array[index[1][0]] - array[index[1][1]]))) / 10
    return deltas

# Function to extract MFCC features from audio
def extract_features(audio, rate):
    mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, nfft=1200, appendEnergy=True)  # Extracting MFCC features
    mfcc_feature = preprocessing.scale(mfcc_feature)  # Scaling the MFCC features
    delta = calculate_delta(mfcc_feature)  # Calculating delta coefficients
    combined = np.hstack((mfcc_feature, delta))  # Combining MFCC features and delta coefficients
    return combined

def record_audio_test():
    FORMAT = pyaudio.paInt16  # Setting audio format
    CHANNELS = 1  # Setting number of audio channels
    RATE = 44100  # Setting audio sampling rate
    CHUNK = 512  # Setting chunk size for audio stream
    RECORD_SECONDS = 10  # Setting duration of recording
    audio = pyaudio.PyAudio()  # Initializing PyAudio object
    print("----------------------record device list---------------------")
    info = audio.get_host_api_info_by_index(0)  # Getting host API info
    numdevices = info.get('deviceCount')  # Getting number of input devices
    for i in range(0, numdevices):
        # Printing input devices available
        if audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
            print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
    print("-------------------------------------------------------------")
    index = -1  # Use default input device
    print("Recording via default input device")
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, input_device_index=index,
                        frames_per_buffer=CHUNK)  # Opening audio stream for recording
    print("Recording started")
    socketio.emit('message', 'Test Recording Started....')  # Emitting a completion message
    Recordframes = []  # Initializing list to store audio frames
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)  # Reading audio data from stream
        Recordframes.append(data)  # Appending audio data to list
    print("Recording stopped")
    socketio.emit('message', 'Test Recording completed.')  # Emitting a completion message
    stream.stop_stream()  # Stopping audio stream
    stream.close()  # Closing audio stream
    audio.terminate()  # Terminating PyAudio object
    OUTPUT_FILENAME = "sample.wav"  # Setting output file name
    
    ### Opened the testing_set_addition.txt file in append mode ('a') and then wrote the OUTPUT_FILENAME to it.
    # WAVE_OUTPUT_FILENAME = os.path.join("testing_set", OUTPUT_FILENAME)  # Generating file path
    # trainedfilelist = open("testing_set_addition.txt", 'a')  # Opening file to store file names
    # trainedfilelist.write(OUTPUT_FILENAME + "\n")  # Writing file name to file list

    WAVE_OUTPUT_FILENAME = os.path.join("testing_set", OUTPUT_FILENAME)  # Generating file path
    with open("testing_set_addition.txt", 'w') as testfilelist:  # Open in 'w' mode to overwrite the file
        testfilelist.write(OUTPUT_FILENAME + "\n")  # Write the filename to the testing set file
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')  # Opening WAV file for writing
    waveFile.setnchannels(CHANNELS)  # Setting number of channels
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))  # Setting sample width
    waveFile.setframerate(RATE)  # Setting frame rate
    waveFile.writeframes(b''.join(Recordframes))  # Writing audio frames to file
    waveFile.close()  # Closing WAV file


# def record_audio_train():
#     Name = request.form['name']  # Get the user's name from the form
#     default_device_index = pyaudio.PyAudio().get_default_input_device_info()['index']  # Get default input device index
#     for count in range(5):  # Recording 5 samples
#         FORMAT = pyaudio.paInt16  # Setting audio format
#         CHANNELS = 1  # Setting number of audio channels
#         RATE = 44100  # Setting audio sampling rate
#         CHUNK = 512  # Setting chunk size for audio stream
#         RECORD_SECONDS = 10  # Setting duration of recording
#         audio = pyaudio.PyAudio()  # Initializing PyAudio object

#         socketio.emit('message', 'Recording sample {} of 5'.format(count + 1))  # Emitting a message to the client

#         stream = audio.open(format=FORMAT, channels=CHANNELS,
#                             rate=RATE, input=True, input_device_index=default_device_index,
#                             frames_per_buffer=CHUNK)  # Opening audio stream for recording

#         Recordframes = []  # Initializing list to store audio frames
#         for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#             data = stream.read(CHUNK)  # Reading audio data from stream
#             Recordframes.append(data)  # Appending audio data to list

#         stream.stop_stream()  # Stopping audio stream
#         stream.close()  # Closing audio stream
#         audio.terminate()  # Terminating PyAudio object

#         OUTPUT_FILENAME = Name + "-sample" + str(count) + ".wav"  # Generating output file name
#         WAVE_OUTPUT_FILENAME = os.path.join("training_set", OUTPUT_FILENAME)  # Generating file path

#         # with open("training_set_addition.txt", 'a') as trainedfilelist:
#         #     trainedfilelist.write(OUTPUT_FILENAME + "\n")  # Writing file name to file list

#         # with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as waveFile:
#         #     waveFile.setnchannels(CHANNELS)  # Setting number of channels
#         #     waveFile.setsampwidth(audio.get_sample_size(FORMAT))  # Setting sample width
#         #     waveFile.setframerate(RATE)  # Setting frame rate
#         #     waveFile.writeframes(b''.join(Recordframes))  # Writing audio frames to file

#          # Upload the recorded audio file to Firebase Storage

#          # this is for uploading all 5 audio to the training_set on firebase storage
#         blob = storage_client.blob('training_set/' + OUTPUT_FILENAME)
#         blob.upload_from_string(b''.join(Recordframes), content_type='audio/wav')

#         # Write the file name to the local file list
#         with open("training_set_addition.txt", 'a') as trainedfilelist:
#             trainedfilelist.write(OUTPUT_FILENAME + "\n")

#         # Download the file list from Firebase Storage
#         blob_list = storage_client.blob('training_set_addition.txt')
#         file_list = blob_list.download_as_string().decode()

#         # Append the new file name to the file list
#         file_list += OUTPUT_FILENAME + "\n"

#         # Upload the updated file list back to Firebase Storage
#         blob_list.upload_from_string(file_list)

#     # Emitting a completion message
#     socketio.emit('message', 'Recording completed for all 5 samples')

def record_audio_train():
    Name = request.form['name']  # Get the user's name from the form
    default_device_index = pyaudio.PyAudio().get_default_input_device_info()['index']  # Get default input device index
    for count in range(5):  # Recording 5 samples
        FORMAT = pyaudio.paInt16  # Setting audio format
        CHANNELS = 1  # Setting number of audio channels
        RATE = 44100  # Setting audio sampling rate
        CHUNK = 512  # Setting chunk size for audio stream
        RECORD_SECONDS = 10  # Setting duration of recording
        audio = pyaudio.PyAudio()  # Initializing PyAudio object

        socketio.emit('message', 'Recording sample {} of 5'.format(count + 1))  # Emitting a message to the client

        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True, input_device_index=default_device_index,
                            frames_per_buffer=CHUNK)  # Opening audio stream for recording

        Recordframes = []  # Initializing list to store audio frames
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)  # Reading audio data from stream
            Recordframes.append(data)  # Appending audio data to list

        stream.stop_stream()  # Stopping audio stream
        stream.close()  # Closing audio stream
        audio.terminate()  # Terminating PyAudio object

        OUTPUT_FILENAME = Name + "-sample" + str(count) + ".wav"  # Generating output file name
        WAVE_OUTPUT_FILENAME = os.path.join("training_set", OUTPUT_FILENAME)  # Generating file path

        with open("training_set_addition.txt", 'a') as trainedfilelist:
            trainedfilelist.write(OUTPUT_FILENAME + "\n")  # Writing file name to file list

        with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as waveFile:
            waveFile.setnchannels(CHANNELS)  # Setting number of channels
            waveFile.setsampwidth(audio.get_sample_size(FORMAT))  # Setting sample width
            waveFile.setframerate(RATE)  # Setting frame rate
            waveFile.writeframes(b''.join(Recordframes))  # Writing audio frames to file

    socketio.emit('message', 'Recording completed for all 5 samples')  # Emitting a completion message


# # Function to train the Gaussian Mixture Model (GMM)
def train_model():
    source = "D:\\SpeakerIdentificationSystem\\website\\training_set\\"  # Setting path to training data
    dest = "D:\\SpeakerIdentificationSystem\\website\\trained_models\\"  # Setting path to save trained models
    train_file = "D:\\SpeakerIdentificationSystem\\website\\training_set_addition.txt"  # Setting path to training file list
    socketio.emit('message', 'Adding Started.....')
    file_paths = open(train_file, 'r')  # Opening training file list
    count = 1  # Initializing count
    features = np.asarray(())  # Initializing array to store features
    for path in file_paths:  # Iterating through each file in the training set
        path = path.strip()  # Removing whitespace characters from both ends of the string
        print(path)
        sr, audio = read(source + path)  # Reading audio file
        print(sr)
        vector = extract_features(audio, sr)  # Extracting features from audio
        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))  # Concatenating features
        if count == 5:  # Training GMM after every 5 samples
            gmm = GaussianMixture(n_components=6, max_iter=200, covariance_type='diag', n_init=3)  # Initializing GMM
            gmm.fit(features)  # Fitting GMM to the features
            picklefile = path.split("-")[0] + ".gmm"  # Generating model file name
            pickle.dump(gmm, open(dest + picklefile, 'wb'))  # Saving trained model to file
            print('+ modeling completed for speaker:', picklefile, " with data point = ", features.shape)
            features = np.asarray(())  # Resetting features
            count = 0  # Resetting count
        count += 1  # Incrementing count
     # Clear training_set_addition.txt
    train_file_path = "D:\\SpeakerIdentificationSystem\\website\\training_set_addition.txt"
    with open(train_file_path, 'w') as file:
        file.write("")
    print("Cleared content of training_set_addition.txt")

    # Delete all recordings from training set
    training_set_path = "D:\\SpeakerIdentificationSystem\\website\\training_set"
    for filename in os.listdir(training_set_path):
        file_path = os.path.join(training_set_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print("Deleted:", file_path)
    socketio.emit('message', 'Added Successfully.....')



def test_model():
    source = "D:\\SpeakerIdentificationSystem\\website\\testing_set\\"  # Setting path to testing data
    modelpath = "D:\\SpeakerIdentificationSystem\\website\\trained_models\\"  # Setting path to trained models
    test_file = "D:\\SpeakerIdentificationSystem\\website\\testing_set_addition.txt"  # Setting path to testing file list
    file_paths = open(test_file, 'r')  # Opening testing file list
    gmm_files = [os.path.join(modelpath, fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]
    models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]  # Loading trained models
    speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname in gmm_files]  # Extracting speaker names
    detected_speakers = []
    threshold = -25  # Set your threshold here

    for path in file_paths:  # Iterating through each file in the testing set
        path = path.strip()  # Removing whitespace characters from both ends of the string
        print(path)
        sr, audio = read(source + path)  # Reading audio file
        vector = extract_features(audio, sr)  # Extracting features from audio
        log_likelihood = np.zeros(len(models))  # Initializing array to store log likelihoods
        for i in range(len(models)):  # Looping through each trained model
            gmm = models[i]  # Getting the GMM for the current speaker
            scores = np.array(gmm.score(vector))  # Getting scores from GMM
            log_likelihood[i] = scores.sum()  # Summing the scores

        max_likelihood = np.max(log_likelihood)
        print(max_likelihood)
        print(threshold)
        if max_likelihood < threshold:
            detected_speakers.append("Not Matched")
        else:
            winner = np.argmax(log_likelihood)  # Getting the index of the maximum score
            detected_speaker = speakers[winner]  # Getting the detected speaker
            detected_speakers.append(detected_speaker)
            print("\tdetected as - ", detected_speaker)  # Printing the detected speaker
        time.sleep(1.0)  # Pausing for 1 second

    return detected_speakers

def get_trained_models():
    model_folder = MODEL_FOLDER  # Use the MODEL_FOLDER variable for the path to the trained models folder
    trained_models = []  # Initialize a list to store trained model names
    
    # Check if the model folder exists
    if os.path.exists(model_folder):
        # Iterate through each file in the model folder
        for file_name in os.listdir(model_folder):
            # Check if the file is a trained model (ends with '.gmm')
            if file_name.endswith('.gmm'):
                # Add the trained model name to the list
                trained_models.append(file_name)
    
    return trained_models



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/record_train', methods=['GET', 'POST'])
def record_train():
    if request.method == 'POST':
        record_audio_train()
        # Instead of returning a plain string, return a JSON response with a message
        return jsonify({'message': 'Audio recorded successfully'})
    return render_template('record_train.html')


@app.route('/train', methods=['GET','POST'])
def train():
    if request.method=='POST':
        # Check if training recordings are available
        train_file_path = "D:\\SpeakerIdentificationSystem\\website\\training_set_addition.txt"
        if not os.path.exists(train_file_path):
            return jsonify({'message': 'No recordings found. Please record your Voice first.'}),400
        # Check if training file is empty
        if os.path.getsize(train_file_path) == 0:
            return jsonify({'message': 'No training recordings found. Please record your Voice first.'}),400
        train_model()
        return jsonify({'message': 'Voice Added successfully'})
    return render_template('record_train.html')

# @app.route('/train', methods=['GET','POST'])
# def train():
#     if request.method == 'POST':
#         # Check if training recordings are available
#         train_file = "training_set_addition.txt"  # Path to training file list in Firebase Storage
#         train_file_path = train_file  # Full path to training file list
#         blob = storage_client.blob(train_file_path)
        
#         if not blob.exists():
#             return jsonify({'message': 'No training recordings found. Please record training samples first.'}), 400

#         # Check if training file is empty
#         if blob.size == 0:
#             return jsonify({'message': 'No training recordings found in training file. Please record training samples first.'}), 400

#         # Start the training process
#         train_model()
#         return jsonify({'message': 'Training Completed successfully'})

#     return render_template('record_train.html')




@app.route('/record_test', methods=['GET', 'POST'])
def record_test():
    if request.method == 'POST':
        record_audio_test()
        return jsonify({'message': 'Audio recorded for testing successfully'})
    return render_template('record_test.html')


@app.route('/test', methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        detected_speakers = test_model()
        return render_template('result.html', speakers=detected_speakers)
    return render_template('record_test.html')


@app.route('/get_models')
def getModel():
    # Get the list of trained models
    models = get_trained_models()
    # Return the list of models as JSON
    return jsonify({'models': models})

@app.route('/dashboard')
def dashboard():
    # Get the list of trained models
    models = get_trained_models()
    # Render the dashboard template and pass the list of trained models to it
    return render_template('dashboard.html', models=models)

@app.route('/rename_model', methods=['POST'])
def rename_model():
    # Get the old and new model names from the request
    old_name = request.args.get('old_name')
    new_name = request.args.get('new_name')

    # Ensure that the new name has the '.gmm' extension
    if not new_name.endswith('.gmm'):
        new_name += '.gmm'

    # Generate file paths for old and new model names
    old_model_path = os.path.join(MODEL_FOLDER, old_name)
    new_model_path = os.path.join(MODEL_FOLDER, new_name)

    try:
        # Rename the model file
        os.rename(old_model_path, new_model_path)
        # Return a success message
        return jsonify({'message': f'Model {old_name} renamed to {new_name} successfully'})
    except FileNotFoundError:
        # If the old model file doesn't exist, return an error message
        return jsonify({'message': f'Model {old_name} not found'}), 404

@app.route('/delete_model', methods=['POST'])
def delete_model():
    # Get the model name to delete from the request
    model_name = request.args.get('model_name')

    # Generate file path for the model to delete
    model_path = os.path.join(MODEL_FOLDER, model_name)

    try:
        # Delete the model file
        os.remove(model_path)
        # Return a success message and the updated list of models
        models = get_trained_models()
        return jsonify({'message': f'Model {model_name} deleted successfully', 'models': models})
    except FileNotFoundError:
        # If the model file doesn't exist, return an error message
        return jsonify({'message': f'Model {model_name} not found'}), 404

@app.route('/signup', methods=['POST'])
def signup():
    # Get user data from the signup form
    first_name = request.form['signupFirstName']
    last_name = request.form['signupLastName']
    email = request.form['signupEmail']
    # password = request.form['password']

    # Store user data in Firestore
    user_ref = db.collection('users').document(email)
    user_ref.set({
        'first_name': first_name,
        'last_name': last_name,
        'email': email
    })

    print("Signup successful")

    # Return a response indicating successful signup
    return jsonify({'message': 'Signup successful'})

@app.route('/user_details/<email>')
def user_details(email):
    # Query Firestore for the user document based on the email
    user_ref = db.collection('users').document(email)
    user_data = user_ref.get().to_dict()

    # Return the user data as JSON response
    return jsonify(user_data)

if __name__ == '__main__':
    app.run(debug=True)