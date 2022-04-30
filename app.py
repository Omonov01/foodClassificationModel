import streamlit as st
from fastai.vision.all import *
import platform
plt = platform.system()
if plt == "Linux": pathlib.WindowsPath = pathlib.PosixPath
import plotly.express as px


st.title("mevalarni klassifikatsiya qiluvchi model")
st.write("yuriqnoma: olma , banan,tort")

#rasmni yuklash uchun joy
file = st.file_uploader("Rasm yuklash",type=["png","jpeg","gif","svg"])
if file:
    img = PILImage.create(file)
    st.image(file)
    #modelni chaqiramiz
    model = load_learner("foods_model.pkl") 
    pred,pred_id,probs = model.predict(img)
    probability = f"{probs[pred_id]*100:.1f}"
    print(probability)
    if 95.0 <= float(probability) <= 100.0:
        st.success(pred)
        st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}")
        fig = px.bar(x=probs*100, y=model.dls.vocab)
        st.plotly_chart(fig)
    else:
        print(st.write("modelimiz to'g'ri ishlashi uchun faqat yuriqnomadagi rasmlarni kiriting"))
        
        
        
