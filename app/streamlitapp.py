import streamlit as st
import os
import imageio
import numpy as np
from PIL import Image

import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

st.set_page_config(layout="wide")

# sidebar
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title("LipBuddy")
    st.info("This application is originally developed from the lipnet deep learning model")

st.title('Lipnet Full Stack App')
#list of options of videos
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns
col1, col2 = st.columns(2)

if options:
    with col1:
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('..', 'data', 's1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering the video
        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)


    with col2:
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        video_np = video.numpy()
        video_np = (video_np * 255).astype(np.uint8)
        frames_gray = [np.squeeze(frame) for frame in video_np]
        frames_gray = [Image.fromarray(frame, 'L') for frame in frames_gray]
        imageio.mimsave('animation.gif', frames_gray, duration=100)
        st.image('animation.gif', width=400)

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)