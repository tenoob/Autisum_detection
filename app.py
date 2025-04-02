import streamlit as st
import numpy as np
#from tensorflow.keras.models import load_model
from io import BytesIO
import tempfile
import tensorflow as tf
import os
import cv2
from collections import Counter
import pickle
from PIL import Image
from rf import load_rf


labels = ['Autistic','Typical']

# To load the model

model1_path = 'model/model.keras'
model2_path = 'model/model (1).keras'
model3_path = 'model/model (2).keras'
model4_path = 'model/model (3).keras'
model5_path = 'model/model (4).keras'
model6_path = 'model/model (5).keras'
model7_path = 'model/model (6).keras'
model8_path = 'model/model (7).keras'
model9_path = 'model/model (8).keras'
model11_path = 'model/model (10).keras'

feature_extractor = 'model/model (9).keras'


model1 = tf.keras.models.load_model(model1_path,compile=False)
model2 = tf.keras.models.load_model(model2_path,compile=False)
model3 = tf.keras.models.load_model(model3_path,compile=False)
model4 = tf.keras.models.load_model(model4_path,compile=False)
model5 = tf.keras.models.load_model(model5_path,compile=False)
model6 = tf.keras.models.load_model(model6_path,compile=False)
model7 = tf.keras.models.load_model(model7_path,compile=False)
model8 = tf.keras.models.load_model(model8_path,compile=False)
model9 = tf.keras.models.load_model(model9_path,compile=False)
model11 = tf.keras.models.load_model(model11_path,compile=False)

feature_extractor = tf.keras.models.load_model(feature_extractor)
rf_model = load_rf()


def class_laabel(prdic,method):
  class_names = ['Autistic','Typical']
  index = np.argmax(prdic)
  class_name = class_names[index]
  score = float(round(prdic.flatten()[index] * 100,2))

  if method == 'class':
    return class_name
  else:
    return class_name,score

def voting(lst):
    counts = Counter(lst)
    most_frequent_element = counts.most_common(1)[0][0]
    return most_frequent_element

def scores(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11):
    c1,s1 = class_laabel(p1,'score')
    c2,s2 = class_laabel(p2,'score')
    c3,s3 = class_laabel(p3,'score')
    c4,s4 = class_laabel(p4,'score')
    c5,s5 = class_laabel(p5,'score')
    c6,s6 = class_laabel(p6,'score')
    c7,s7 = class_laabel(p7,'score')
    c8,s8 = class_laabel(p8,'score')
    c9,s9 = class_laabel(p9,'score')
    c10,s10 = class_laabel(p10,'score')
    c11,s11 = class_laabel(p11,'score')

    score = f"""Model 1 : {c1} : {s1}
    Model 2 : {c2} : {s2}
    Model 3 : {c3} : {s3}
    Model 4 : {c4} : {s4}
    Model 5 : {c5} : {s5}
    Model 6 : {c6} : {s6}
    Model 7 : {c7} :  {s7}
    Model 8 : {c8} :  {s8}
    Model 9 : {c9} :  {s9}
    Model 10 : {c10} : {s10}
    Model 11 : {c11} : {s11}
    """
    return score

def make_prediction(input_data):

    #input_data = input_data.reshape(1, 224, 224, 3)

    predict1 = model1.predict(input_data)
    predict2 = model2.predict(input_data)
    predict3 = model3.predict(input_data)
    predict4 = model4.predict(input_data)
    predict5 = model5.predict(tf.image.rgb_to_grayscale(input_data))
    predict6 = model6.predict(input_data)
    predict7 = model7.predict(input_data)
    predict8 = model8.predict(input_data)
    predict9 = model9.predict(input_data)
    
    featur = feature_extractor.predict(input_data)
    features = featur.reshape(featur.shape[0], -1)
    predict10 = rf_model.predict(features)
    #predict10 = 'Autistic'
    predict11 = model11.predict(tf.image.rgb_to_grayscale(input_data))

    mylst = []
    mylst.append(class_laabel(predict1,'class'))
    mylst.append(class_laabel(predict2,'class'))
    mylst.append(class_laabel(predict3,'class'))
    mylst.append(class_laabel(predict4,'class'))
    mylst.append(class_laabel(predict5,'class'))
    mylst.append(class_laabel(predict6,'class'))
    mylst.append(class_laabel(predict7,'class'))
    mylst.append(class_laabel(predict8,'class'))
    mylst.append(class_laabel(predict9,'class'))
    mylst.append(class_laabel(predict10,'class'))
    mylst.append(class_laabel(predict11,'class'))

    print(mylst)
    result = voting(mylst)
    print("final prediction:",result)

    score = scores(predict1,predict2,predict3,predict4,predict5,predict6,predict7,predict8,predict9,predict10,predict11)

    return result,score



def load_image(image_file):
    """Loads an image from a file-like object and returns a NumPy array."""
    try:
        img = Image.open(image_file)
        return np.array(img)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def img_process(mri):
  #input_data = cv2.imread(mri, cv2.IMREAD_COLOR)
  input_data = cv2.resize(mri, (224, 224))
  input_data = cv2.cvtColor(input_data, cv2.COLOR_RGB2BGR)
  input_data = input_data / 255.0
  input_data = input_data.reshape(1,224,224,3)
  return input_data





# Set the title of the app
st.title("🧠 Autism Spectrum Disorder (ASD) Detection")
st.write("Upload a Brain MRI image to predict if the subject has ASD.")

# Upload Image
uploaded_file = st.file_uploader("📤 Choose an MRI image...", type=["jpg", "jpeg", "png"])
print(uploaded_file)

if uploaded_file is not None:

    img_array = load_image(uploaded_file)
    input_data = img_process(img_array)
    
    if img_array is not None and img_array.size > 0:
        st.image(img_array, caption="Uploaded Image.", use_container_width=True)
        processed_img = img_process(img_array.copy()) 

    print(processed_img.shape)
    prediction,score = make_prediction(input_data)
    print(f"predictions: { np.argmax(prediction)}")
    #st.write(f"predictions: { prediction}")

    # Display Prediction
    st.markdown(f"""
    <div style="background-color: rgba(0, 123, 255, 0.3); color: black; padding: 20px; 
                border-radius: 10px; text-align: center; font-size: 24px; margin-bottom: 30px;">
         Predicted Class: {prediction}
    </div>
    """, unsafe_allow_html=True)


    f"""# Score
    Models : Class : Score
    {score}
    """

    
