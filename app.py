import streamlit as st
from streamlit import session_state
from streamlit_option_menu import option_menu

import matplotlib.pyplot as plt
import numpy as np
import cv2
\


# set page layout to wide
st.set_page_config(layout="wide")
# upload css file

st.markdown("", unsafe_allow_html=True)


def body():
    # dividing the page to 2 parts for output and input image
    col1, col2 = st.columns(2)
    # side bar that contains a select box for choosing your page option
    with st.sidebar:
        which_page = st.selectbox(
            "Choose Page", ["Optimal Thresholding", "Otsu Thresholding", "Spectral Thresholding","Local Thresholding",
                            "K-Means Segmentation","Region Growing Segmentation","Agglomerative Segmentation","Mean Shift Segmentation"]
        )
        # uploading the file browser
        file = st.file_uploader("Upload file", type=["jpg", "png"])
        segmentation_pages = False
        if(which_page=='K-Means Segmentation' or which_page=="Region Growing Segmentation" or which_page =="Agglomerative Segmentation" or which_page=='Mean Shift Segmentation'):
            segmentation_pages = True
            
    if file:
        # option to choose between the gray scale or the RGB
        with st.sidebar:
            if(segmentation_pages):
                luv_checkbox = st.checkbox(key="luv_checkbox", label="LUV")
                num_clusters = st.slider("Number Of Clusters", min_value=2,max_value=10, value=5)
                threshold = st.slider("Threshold")
                max_iterations=st.slider("Maximum iterations", min_value=2,max_value=100, value=5)

        with col1:  # first part of the image for displaying the original image
            st.header("Original Image")
            # here we made a specific location for uploading the images and it is the relative folder images
            img_original = cv2.imread(file.name)
            img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
            st.image(img_original, use_column_width=True)

        # the first option for the filter page which contains a lot of different filters
        if which_page == "Optimal Thresholding":
            print("call function")
        elif which_page == "Otsu Thresholding":
            print("call")
        elif which_page=="Spectral Thresholding":
            print("call")
        elif which_page=="Local Thresholding":
            print("call")
        elif which_page=="K-Means Segmentation":
            print("call")
        elif which_page=="Region Growing Segmentation":
            print("call")
        elif which_page=="Agglomerative Segmentation":
            print("call")

        else:
            #Mean Shift Segmentation
            print("call")
        with col2:
            st.header("Output Image")
            output_img=cv2.imread("output.png")
            output_img=cv2.cvtColor(output_img,cv2.COLOR_BGR2RGB)
            st.image(output_img, use_column_width=True)


if __name__ == "__main__":
    body()