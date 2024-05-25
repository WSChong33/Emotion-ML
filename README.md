# Emotion Detection

- Training a CNN model to detect real-time facial emotion

# List of technologies:
- Python
- PyTorch
- OpenCV
- pandas
- NumPy
- Python Imaging Library

- Setup:
    - Run "python3 -m venv path/to/venv"
    - Run "source path/to/venv/bin/activate"
    - Run "pip3 install -r requirements.txt"

- Obtain dataset:
    - Run "kaggle competitions download -c challenges-in-representation-learning-facial-expression-recognition-challenge"
    - Run "unzip challenges-in-representation-learning-facial-expression-recognition-challenge.zip"
    - Run "tar -xzvf fer2013.tar.gz"
    - Run "mv fer2013/fer2013.csv dataset.csv"
    - (Optional): Delete all unnecessary files/directory - fer2013, .zip files, all other .csv files, .gz files
    

- Train model + Test:
    - Run "cd src" to change working directory
    - Run "python3 data_to_img.py" to convert CSV data in dataset.csv to image (PNG files) -> This will generate a dataset directory in the parent directory
    - Run "python3 train.py" to train model
    - Run "python3 real_time_emotion_detection.py" to run program