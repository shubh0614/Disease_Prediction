import streamlit as st
import numpy as np
import sklearn
import pickle
from PIL import Image

img = Image.open("img.jpeg")
st.image(img)
st.header('Disease Predictor', divider='rainbow')
COLUMNS = [i.strip("\n") for i in open("Symptoms.txt", "r").readlines()]
options = st.multiselect('What are your symptoms', COLUMNS)

if st.button("Submit"):
  input_val = [1 if i in options else 0 for i in COLUMNS]
  print(input_val)

  input_val = np.array(input_val).reshape(1, -1)
  model = pickle.load(open("model_RF.pkl", "rb"))
  predicted_val = model.predict(input_val)
  st.write("The Predicted Disease is: " + predicted_val[0])