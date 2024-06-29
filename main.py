import streamlit as st
from ultralytics import YOLO
from assets import process_video, process_image
from PIL import Image
import cv2

PROCESS = "Processing..."
model = YOLO("model\model_helmet.pt")

st.title("YOLOv10 for helmet detection! 👷")
st.markdown("##### By Nguyen Anh Duy")
with st.sidebar:
    st.header("Upload your image or video")
    uploaded_file = st.file_uploader("Choose an image or video", type=[
                                     "jpg", "jpeg", "png", "mp4"])
    st.header("Or view our demo")
    button_image = st.button("Demo Image")
    button_video = st.button("Demo Video")


if uploaded_file is not None:
    if uploaded_file.type == "video/mp4":
        st.video(uploaded_file)
        # progres bar
        with st.spinner(PROCESS):
            process_video(uploaded_file, model)
    else:
        st.image(uploaded_file)
        with st.spinner(PROCESS):
            process_image(uploaded_file, model)

elif button_image:
    image = Image.open("demo\helmet.jpg")
    st.image(image)
    with st.spinner(PROCESS):
        process_image('demo\helmet.jpg', model)

elif button_video:

    st.video("demo\helmet.mp4")
    with st.spinner(PROCESS):
        process_video("demo\helmet.mp4", model)
