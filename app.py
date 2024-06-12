# Import convention
import streamlit as st
import pandas as pd
import glob
import PIL
import os
from PIL import Image
from utils.pipeline import thyroid_image_classification

st.header('Thytra (Thyroid Ultrasound Image Classification for Disease Diagnosis)')
st.info("การใช้งาน เลือกไฟล์รูปภาพอัลตราซาวนด์ต่อมไทรอยด์ โมเดลจะทำการ predict ว่ามีโอกาสเป็น Benign , Malignant , Normal กี่ %")

@st.cache_resource
def load_classifier(model_name):
    return thyroid_image_classification(model_name=model_name)

classifier = thyroid_image_classification(model_name="agent593/Thyroid-Ultrasound-Image-Classification-ViTModel")

st.sidebar.image('image/logo.jpg')
st.sidebar.write('Ai Builders ปีที่ 4 กลุ่ม loyal-coyotes')
st.sidebar.write('จัดทำโดย บุณยาพร ชัยมงคลทรัพย์')
st.sidebar.write('### Enter image to classify')

option = st.sidebar.radio('', ['Use a validation image', 'Use your own image'])
valid_images = glob.glob('data/test/*')
valid_images.sort()

if option == 'Use a validation image':
    st.sidebar.write('### Select a validation image')
    filenames = [os.path.basename(fname) for fname in valid_images]  # Get only the filenames
    selected_filename = st.sidebar.selectbox('', filenames)
    if selected_filename:
        image_path = os.path.join('data/test', selected_filename)  # Reconstruct the full path
        image = Image.open(image_path)
        st.sidebar.image(image)
        st.session_state['result'] = image

else:
    st.sidebar.write('### Select an image to upload')
    fname = st.sidebar.file_uploader('',
                                     type=['png', 'jpg', 'jpeg'],
                                     accept_multiple_files=False)
    if fname:
        image = Image.open(fname)
        st.sidebar.image(image)
        st.session_state['result'] = image

if 'result' in st.session_state and st.session_state['result'] is not None:
    st.image(st.session_state['result'])
    # Get classification results
    classification_results = classifier(st.session_state['result'])
    # Find the highest score label
    max_score_prediction = max(classification_results, key=lambda x: x['score'])
    max_score_label = max_score_prediction['label']
    # Display the highest score label
    st.text(classifier(st.session_state['result']))
    st.text(f"Prediction: {max_score_label} ({max_score_prediction['score']*100:.2f}%)")
    
st.sidebar.image('image/aib.png')
st.sidebar.write('โครงการ AI Builders จัดขึ้นจากความร่วมมือระหว่าง VISTEC, AIResearch, Central Retail Digital และมหาวิทยาลัยมหิดลเพื่อพัฒนาองค์ความรู้ทางด้านวิทยาศาสตร์ข้อมูล (Data Science) และปัญญาประดิษฐ์ (Artificial Intelligence / AI) ให้กับน้องๆระดับมัธยมต้น-ปลาย ที่สนใจอยากเรียนรู้และพัฒนาโครงงานที่ใช้ทักษะในด้านนี้เพื่อประยุกต์ใช้จริงในชีวิตประจำวัน ในปี 2024 เราได้รับการสนับสนุนจาก บพค., AWS, DELL, VISAI, กลุ่ม OSK Artificial Intelligence, และ Krungsri Nimble')