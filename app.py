import streamlit as st
from streamlit import session_state
from streamlit_option_menu import option_menu
import thresholding as th
from luv import RGB2LUV
import matplotlib.pyplot as plt
from segmentationLUV import *
from segmentationRGB import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segmantation_using_region_growing import *



# set page layout to wide
st.set_page_config(layout="wide")
# upload css file

st.markdown("", unsafe_allow_html=True)


def body():
    # side bar that contains a select box for choosing your page option
    with st.sidebar:
        which_page = st.selectbox(
            "Choose Page", ["Optimal Thresholding", "Otsu Thresholding", "Spectral Thresholding",
                            "K-Means Segmentation","Region Growing Segmentation","Agglomerative Segmentation","Mean Shift Segmentation"]
        )
        # uploading the file browser
        file = st.file_uploader("Upload file", type=["jpg", "png"])
        segmentation_pages = False
        if(which_page=='K-Means Segmentation' or which_page=="Region Growing Segmentation" or which_page =="Agglomerative Segmentation" or which_page=='Mean Shift Segmentation'):
            segmentation_pages = True

    if(not segmentation_pages):
        col1,col2,col3=st.columns(3)
    else :
        col1, col2 = st.columns(2)

    if file:
        # option to choose between the gray scale or the RGB
        if which_page != "Region Growing Segmentation":
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
            if not segmentation_pages:
                img_original=cv2.resize(img_original,(300,300))
            img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
            st.image(img_original, use_column_width=True)

        # the first option for the filter page which contains a lot of different filters
        thresholdType=["Optimal","Otsu","spect"]
        optionIndex=0
        if which_page == "Optimal Thresholding":
            optionIndex=0
        elif which_page == "Otsu Thresholding":
            optionIndex=1
        elif which_page=="Spectral Thresholding":
            optionIndex=2
        elif which_page=="K-Means Segmentation":
            if luv_checkbox:
                output_img, labels=kmeans(img_original, k=5, max_iter=50, luv=True)

            else:
                output_img, labels=kmeans(img_original, k=5, max_iter=50, luv=False)

        elif which_page=="Region Growing Segmentation":
            seeds = [[200, 300], [300, 295], [310, 350]]
            segment_image_class = RegionGrower(cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY), seeds, 6)
            output_img = segment_image_class.fit()
        elif which_page=="Agglomerative Segmentation":
            if luv_checkbox:
                output_img=agglomerative(img_original, clusters_numbers=5 , luv=True)

            else:
                output_img=agglomerative(img_original,clusters_numbers=5,  luv=False)
        else:
            #Mean Shift Segmentation
            if luv_checkbox:
                output_img = mean_shift(img_original, threshold=30, luv=True)

            else:
                output_img = mean_shift(img_original, threshold=30, luv=False)
                
        if (not segmentation_pages):
            th.Global_threshold(img_original.copy(),
                                thresholdType[optionIndex])

        with col2:
            st.header("Output Image")
            output_img=cv2.imread("output.png")
            output_img=cv2.cvtColor(output_img,cv2.COLOR_BGR2RGB)
            st.image(output_img, use_column_width=True)
        if(not segmentation_pages):
            with col3:
                st.header("local thresolding ")
                st.sidebar.header("please wait local thresholding takes about 3-5 minutes ☺")
                th.Local_threshold(img_original,thresh_typ=thresholdType[optionIndex])
                st.image("outputLocal.png",use_column_width=True)

if __name__ == "__main__":
    body()