from segmentation import KMeans
from segmentation import AgglomerativeClustering
from segmentation import MeanShift
# from segmentation import regionGrow
# from segmentation import Point
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from luv import RGB2LUV

def kmeans(image, k=5, max_iter=100, luv=False):
    img=np.copy(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if(luv):
        img = RGB2LUV(img)
    pixel_vals = img.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)
    model = KMeans(k, max_iters=max_iter)
    y_pred = model.predict(pixel_vals)
    centers = np.uint8(model.cent())
    y_pred = y_pred.astype(int)
    # flatten labels and get segmented image
    labels = y_pred.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(img.shape)
    saved=mpimg.imsave("output.png", segmented_image)
    return segmented_image, labels

def agglomerative(image, clusters_numbers=2, initial_clusters= 25, luv=False):
    src = np.copy(image)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    if(luv):
        print("LUVVVV")
        src = RGB2LUV(src)
    agglomerative = AgglomerativeClustering(image=src, clusters_numbers=clusters_numbers,
                                            initial_k=initial_clusters)
    saved=mpimg.imsave("output.png", agglomerative.output_image)
    return agglomerative.output_image

def mean_shift(image, threshold= 60, luv=False):
    src = np.copy(image)
    # convert to RGB
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    if(luv):
        src = RGB2LUV(src)

    ms = MeanShift(image=src, threshold=threshold)
    ms.run_mean_shift()
    output = ms.get_output()
    saved=mpimg.imsave("output.png", output)

    return output

# def apply_region_growing(image: np.ndarray,threshold):

#     src = np.copy(image)
    
#     src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
#     src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
#     seeds = []

#     for i in range(3):
#         #select random x and y as an initial seed(it will differ from each run)
#         x = np.random.randint(0, src.shape[0])
#         y = np.random.randint(0, src.shape[1])
#         seeds.append(Point(x, y))
#         # seeds = [Point(10, 1), Point(5, 15), Point(100, 150)] # if we chose constant points for each run

#     output_image = regionGrow(src, seeds, threshold)
#     saved=mpimg.imsave("region_grow.png", output_image,cmap="gray")

#     return output_image

if __name__ == "__main__":

    img = cv2.imread('snf.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    segmentedimg= agglomerative(image=img, clusters_numbers=5,luv=False)
    cv2.imshow('image' , img)
    output = cv2.imread('output.png')

    cv2.imshow('output' ,  output )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
