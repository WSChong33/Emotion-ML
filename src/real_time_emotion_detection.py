import cv2
import torch
import numpy as np
from torchvision import transforms
from model import EmotionCNN
from PIL import Image

# Load the trained model
model = EmotionCNN()
model.load_state_dict(torch.load('emotion_cnn.pth'))
model.eval()

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Define the transformation
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Initialize the webcam
cap = cv2.VideoCapture(cv2.CAP_AVFOUNDATION + 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi_pil = Image.fromarray(roi)  # Convert NumPy array to PIL Image
        roi_transformed = transform(roi_pil).unsqueeze(0)

        with torch.no_grad():
            output = model(roi_transformed)
            _, predicted = torch.max(output, 1)
            emotion = emotion_labels[predicted.item()]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Emotion Detector', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
