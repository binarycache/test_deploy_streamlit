import streamlit as st
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import os
# import the necessary packages for image recognition
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import Xception  # TensorFlow ONLY
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import cv2
from PIL import Image
from io import BytesIO
import pandas as pd
import urllib
from classify import *
plt.style.use("ggplot")

# set page layout
st.set_page_config(
    page_title="Image Classification App",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Welcome to CogXRLabs!")

page = st.sidebar.radio("Navigation",["Home", "Register", "Applications"])
logged_in = False

if os.path.exists("user_database.py"):
    user_dict = np.read("user_database.np")
else:
    user_dict = defaultdict(list)


if page=="Home":
    st.warning("If you are visiting for the first time, please consider registeing yourself. You can register yourself from choosing the appropriate action on the left sidebar")
    st.markdown(""" ## About us <br>
    CogXRLabs is a For-profit Social Enterprise with an aim to commence Healthcare 4.0 in India. It is a confluence of state-of-the-art cutting-edge technologies including, but not limited to, AI (Machine Learning and Deep Learning), Extended Reality (Virtual, Augmented and Mixed), and Brain Computer Interfaces. <br>
    ## Vision <br>
    Our vision is to introduce improved, transformational suite of products and expert services that address the current challenges faced by the healthcare industry in India. The aim is to commence the era of Healthcare 4.0 by providing patients with better, more value-added, and more cost-effective healthcare services while also improving the industry's efficacy and efficiency.
    """,True)
    
    
elif page=="Register":
    #st.set_page_config("Register", layout='centered',page_icon = ":clipboard:")

    st.title("Registration Form")



    first_name, last_name = st.columns(2)
    first_name.text_input("First Name")
    last_name.text_input("Last Name")
    
<<<<<<< HEAD
    age, gender = st.columns([3,1])
=======
    age, gender, = st.columns([3,1])
>>>>>>> 703f9b07132fcc7bea2394199cf0a99188d7f391
    age.number_input("Age (in Years)",min_value = 1,max_value=100,value = 30,step = 1)
    gender.selectbox("Gender",["Male","Female","Others"],index = 0)

    email_id, mobile_number = st.columns([2,2])
    email_id.text_input("Email ID")
    mobile_number.text_input("Mobile Number")

    col5 ,col6 ,col7  = st.columns(3)
    username = col5.text_input("Username")
    password =col6.text_input("Password", type = "password")
    col7.text_input("Repeat Password" , type = "password")

    but1,but2,but3 = st.columns([1,5,1])

    agree  = but1.checkbox("I Agree")

    if but3.button("Submit"):
        if agree: 
            user_dict["first_name"].append(first_name)
            user_dict["last_name"].append(last_name)
            user_dict["age"].append(age)
            user_dict["gender"].append(gender)
            user_dict["username"].append(username)
            user_dict["password"].append(password)
            user_dict["email_id"].append(email_id)
            user_dict["mobile_number"].append(mobile_number)
            st.success("Done")
        else:
            st.warning("Please Check the T&C box")

elif page=="Applications":
    choice = st.selectbox("Choose one of the following Applications",["Covid Classifier", "Breast Cancer Classifier"], index=0)
    image = st.file_uploader(f"Please Choose an X-Ray Image for {choice.split(' ')[0]} classification", ['jpg', 'jpeg','png'])

    if image:
<<<<<<< HEAD
        _, input_image,_ = st.columns([2,3,1])
        input_image.image(image, width=300)
        img = Image.open(image).convert('RGB') # 3 channels
        st.write("")
        with st.spinner(text="Classifying now..."):
            prediction = classify(img, 'mobile_netv2.h5')
        _, label,_ = st.columns([2,3,1])
        if prediction == 0:
            label.header("X-ray image has Covid")
        elif prediction==1:
            label.header("X-Ray scan is healthy")
        else:
            label.header("X-Ray has pneumonia")
=======
        st.image(image)
        img = Image.open(image).convert('RGB') # 3 channels
        st.write("")
        st.write("Classifying now...")
        label = classify(img, 'vgg.h5')
        if label == 0:
            st.header("X-ray image has Covid")
        elif label==1:
            st.header("X-Ray scan is healthy")
        else:
            st.header("X-Ray has pneumonia")
>>>>>>> 703f9b07132fcc7bea2394199cf0a99188d7f391
        # st.subheader(f"Top Predictions from {network}")
        # st.dataframe(
        #     pd.DataFrame(
        #         predictions[0], columns=["Network", "Classification", "Confidence"]
        #     )
        # )
<<<<<<< HEAD
=======
 

>>>>>>> 703f9b07132fcc7bea2394199cf0a99188d7f391
st.caption("Made with 💟 by CogXRLabs.")