import cv2
import tempfile
import streamlit as st
from PIL import Image


def process_video(uploaded_file, model):

    if isinstance(uploaded_file, str):
        cap = cv2.VideoCapture(uploaded_file)
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

    # Create a Streamlit video player
    video_placeholder = st.empty()

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Convert BGR to RGB
            annotated_frame_rgb = cv2.cvtColor(
                annotated_frame, cv2.COLOR_BGR2RGB)

            # Display the annotated frame
            video_placeholder.image(
                annotated_frame_rgb, channels="RGB", use_column_width=True)
        else:
            break

    cap.release()


def process_image(uploaded_file, model):
    # Load the image
    image = Image.open(uploaded_file)

    # Run YOLOv8 inference on the image
    results = model(image)

    # Visualize the results on the image
    annotated_image = results[0].plot()

    # Convert BGR to RGB
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Display the annotated image
    st.image(annotated_image_rgb, channels="RGB", use_column_width=True)
