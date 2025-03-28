# Face Mask Detection Model (Along with Demo Video)

This project leverages an Image Processing model to create a classifier system for detecting whether a person is wearing a face mask using OpenCV and Haar cascades.

## Table of Contents

- [Introduction](#introduction)
- [Demo](#demo)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

The Face Mask Detection Model is a real-time application that uses computer vision techniques to identify whether a person is wearing a mask or not. It utilizes the OpenCV library and pre-trained Haar cascades for face and eye detection.
## Demo
https://github.com/gpra012333/Face-Mask-Detection/assets/142736928/a632400e-6c89-4d4d-b9cf-28fd86d3daed
## Features

- Real-time face detection using a webcam.
- Classification of individuals as wearing a mask or not wearing a mask based on the presence of eyes and face.
- Visual indication on the screen showing the detection result.

## Installation

### Prerequisites

- Python 3.11.5
- OpenCV library

### Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/gpra012333/face-mask-detection.git
    cd face-mask-detection
    ```

2. Install the required packages:
    ```bash
    pip install opencv-python
    ```

3. Download the Haar cascade XML files for face and eye detection:
    - [haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
    - [haarcascade_eye.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_eye.xml)

4. Place the XML files in the same directory as the script.

## Usage

1. Run the Python script:
    ```bash
    python app.py
    ```

2. The application will start the webcam and begin detecting faces and eyes in real-time. Based on the detection results, it will display one of the following messages on the screen:
    - "Person Wearing Mask" (if no face is detected but eyes are detected)
    - "Person not Wearing Mask" (if a face is detected along with eyes)

### Code Explanation

```python
import cv2

def get_predict():   
    cap = cv2.VideoCapture(0)  # Start the webcam
    while True:
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            print("Failed to capture image")
            break

        # Load Haar cascades for face and eye detection
        face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)  # Detect faces
        eyes = eye_classifier.detectMultiScale(gray)  # Detect eyes

        if len(eyes) != 0:
            if len(faces) == 0:
                cv2.putText(frame, "Person Wearing Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Person not Wearing Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow('Face', frame)  # Display the resulting frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Exit on pressing 'q'

    cap.release()
    cv2.destroyAllWindows()

get_predict()
