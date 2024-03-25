import streamlit as st
from fastai.vision.all import *
import plotly.express as px

#title
st.title("This is project which indentify Sport Equipments like Parachute, training bench, Snowboard, Ski, Lifejacket, Punching bag, and Surfboard")


#Image uploading
file = st.file_uploader("Rasm yuklash", type = ['png','jpeg','gif','svg','jpg','webp'])

if file:
    st.image(file)
    #PIL convert
    img = PILImage.create(file)

    #model
    model = load_learner("Sport_equipment_classifier.pkl")

    #Prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat:{pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

    #Plotting
    fig = px.bar(x=probs*100, y = model.dls.vocab)
    st.plotly_chart(fig)