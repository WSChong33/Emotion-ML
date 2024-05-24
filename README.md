# Emotion Detection

- Machine learning algorithm to detect real-time facial emotion using camera

- To obtain dataset:
    - Run "kaggle competitions download -c challenges-in-representation-learning-facial-expression-recognition-challenge"

- To initialise:
    - Run "cd src" to change working directory
    - Run "python3 data_to_img.py" to convert CSV data in dataset.csv to image (PNG files) -> This will generate a dataset directory in the parent directory
    - Run "python3 train.py" to train model
    - Run "python3 real_time_emotion_detection.py" to run program