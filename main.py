import tensorflow as tf 
import numpy as np 

import streamlit as st
import numpy as np 

model=tf.keras.models.load_model("generator5.h5",compile=False)





with st.sidebar:
    st.title("Face Generator")
    choice=st.radio("Navigation",["Generate Image","Details"])
    st.info("this is a DCGAN model trained on celeb_a dataset")

st.title("FACE GENERATOR")
def make_image():
    global col1,col2
    
    y=tf.random.normal([1,100])
    z=np.array([y])
    z=z.reshape(10,10,1)
    z=tf.keras.utils.array_to_img(z)
    img=model.predict(y)
    for i in img:
       x= tf.keras.utils.array_to_img(i)
    
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)
    
    col1.header("INPUT IMAGE")
    col2.header("OUTPUT IMAGE")
    col1.image(z)
    col2.image(x)


if choice=="Generate Image":
    
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)
    st.button( label="GENERATE IMAGE",on_click=make_image)
with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)
col1,col2=st.columns(2)
if choice=="Code Of The Model":
    st.title("INFORMATION ABOUT MODEL")
    st.write("""The pacakages this project used as tensorflow,numpy,streamlit. The Model has a custum DCGAN architecture.
    It is trained on the celeb_a dataset .
    """)