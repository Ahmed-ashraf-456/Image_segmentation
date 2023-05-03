from segmentation import KMeans
from segmentation import AgglomerativeClustering
from segmentation import MeanShift
from segmentation import regionGrow
from segmentation import Point
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def apply_k_means_rgb(source, k=5, max_iter=100):
    
    # convert to RGB
    img=np.copy(source)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

    # reshape image to points
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # run clusters_num-means algorithm
    model = KMeans(k, max_iters=max_iter)
    y_pred = model.predict(pixel_values)

    centers = np.uint8(model.cent())
    y_pred = y_pred.astype(int)

    # flatten labels and get segmented image
    labels = y_pred.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(img.shape)
    saved=mpimg.imsave("kmeans_rgb.png", segmented_image)

    return segmented_image, labels



def apply_mean_shift_rgb(source: np.ndarray, threshold: int = 60):
    

    src = np.copy(source)

    # convert to RGB
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    
    

    ms = MeanShift(source=src, threshold=threshold)
    ms.run_mean_shift()
    output = ms.get_output()
    saved=mpimg.imsave("meanshift_rgb.png", output)

    return output



if __name__ == "__main__":

    img = cv2.imread('seg-image.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # segmentedimg,labels= apply_k_means_rgb(source=img)
    segmentedimg,labels= apply_agglomerative_rgb(source=img)

    plt.figure()

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title('Original image')

    plt.subplot(1, 2, 2)
    plt.imshow(segmentedimg,cmap="gray")
    plt.axis('off')
    plt.title(f'Segmented image')

    plt.show()