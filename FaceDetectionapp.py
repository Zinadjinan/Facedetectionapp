import cv2
import streamlit as st
import os

# Function to detect faces from the webcam frames
def detect_faces(rectangle_color, min_neighbors, scale_factor):
    # Create a VideoCapture object to capture video from the webcam
    video_capture = cv2.VideoCapture(0)

    # Create a CascadeClassifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        # Read the video frame
        ret, frame = video_capture.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(30, 30))

        # Draw bounding boxes around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), rectangle_color, 2)

        # Display the frame with detected faces
        cv2.imshow('Face Detection', frame)

        # Save the frame with detected faces
        save_image(frame, faces)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture and destroy the OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

# Function to save the image with detected faces
def save_image(frame, faces):
    if len(faces) > 0:
        # Create a folder to save the images if it doesn't exist
        folder_path = 'detected_faces'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save the image with bounding boxes
        image_with_faces = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(image_with_faces, (x, y), (x+w, y+h), rectangle_color, 2)

        # Generate a unique filename
        filename = os.path.join(folder_path, f'detected_face_{len(os.listdir(folder_path)) + 1}.jpg')

        # Save the image
        cv2.imwrite(filename, image_with_faces)

# Streamlit app interface
st.title("Face Detection from Webcam")

# Instructions
st.markdown("## Instructions:")
st.markdown("Click the 'Start' button to detect faces from the webcam.")
st.markdown("The webcam feed will open in a separate window, and faces will be outlined with bounding boxes.")
st.markdown("To stop the face detection, close the webcam window or press the 'q' key.")
st.markdown("Detected images will be saved in a folder named 'detected_faces'.")

# User-defined color for the rectangles
rectangle_color = st.color_picker("Select the color for the rectangles")

# User-defined minNeighbors parameter
min_neighbors = st.slider("Adjust the minNeighbors parameter", min_value=1, max_value=10, value=5)

# User-defined scaleFactor parameter
scale_factor = st.slider("Adjust the scaleFactor parameter", min_value=1.1, max_value=2.0, value=1.1, step=0.1)

# Start face detection when the user clicks the 'Start' button
if st.button('Start'):
    # Call the detect_faces function with the chosen color, minNeighbors, and scaleFactor values
    detect_faces(rectangle_color, min_neighbors, scale_factor)

