import os
import glob
import streamlit as st
from PIL import Image
import numpy as np
import torch
from datetime import datetime

st.title("Peach Counter")
st.write("開発途中ですが、果物の着果量を確認することを目指しています")

pages = dict(
    page1="normal",
    page2="grayscale",
    page3="prediction-yolov5s",
    page4="prediction-peach2311"
)

outputpath = r"data/results"
uploaded_img = st.sidebar.file_uploader("画像ファイルをアップロードしてください", type=["jpg", "jpeg", "png"])
img = Image.open(r"data/samples/peach20220703.jpg")
img_array = np.array(img)
gray_array = np.array(img.convert("L"))
yolov5s_array = np.array(Image.open(os.path.join(outputpath, "yolov5s_default.jpg")))
peach_array = np.array(Image.open(os.path.join(outputpath, "peach2311_default.jpg")))

print(img_array)

def yolov5s_pred(input):

    #Torch Hubからyolov5をダウンロード
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    img = Image.open(imgpath)
    result = model(img)
    result.render()
    for im in result.ims:
        im_base64 = Image.fromarray(im)
        im_base64.save(os.path.join(outputpath, "yolov5s.jpg"))
          

def peach2311_pred(input):

    #独自モデルを使用
    model = torch.hub.load('sub', 'custom', 'peach2311.pt', source='local')
    img = Image.open(imgpath)
    result = model(img)
    result.render()
    for im in result.ims:
        im_base64 = Image.fromarray(im)
        im_base64.save(os.path.join(outputpath, "peach2311.jpg"))
          

if uploaded_img:
    st.sidebar.write("処理を選んでください")
    img = None
    img_array = None
    im = Image.open(uploaded_img)
    ts = datetime.timestamp(datetime.now())
    imgpath = os.path.join('data/uploads', "uploaded.jpg")
    im.save(imgpath)
    img_array = np.array(im)
    gray_array = np.array(im.convert("L"))
    yolov5s_pred(uploaded_img)
    peach2311_pred(uploaded_img)
    yolov5s_array = np.array(Image.open(os.path.join(outputpath, "yolov5s.jpg")))
    peach_array = np.array(Image.open(os.path.join(outputpath, "peach2311.jpg")))
    
page_id = st.sidebar.selectbox(
    "画像処理を選択",
    ["page1", "page2", "page3", "page4"],
    format_func=lambda page_id: pages[page_id], 
    key="page-select"
)

def page1(): 
    st.write("normal")
    st.image(img_array, caption="image", use_column_width=True) 
        
def page2():
    st.write("grayscale")
    st.image(gray_array, caption="grayscale", use_column_width=True)

def page3():
    st.write("prediction-yolov5s")
    st.image(yolov5s_array, caption="prediction", use_column_width=True)

def page4():
    st.write("prediction-peach2311")
    st.image(peach_array, caption="prediction", use_column_width=True)

if page_id == "page1":
    page1()

if page_id == "page2":
    page2()

if page_id == "page3":
    page3()

if page_id == "page4":
    page4()