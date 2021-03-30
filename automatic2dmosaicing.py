import cv2 as cv
import imutils
import sys
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt

# ALGORITHM
# 1) Choose one image of the set as a reference
# 2) Compute the homography H which maps one of the other images of the set to this reference image
# 3) Projectively warp the image with this homography, and augment the reference image with the non-overlapping part of
#    warped image
# 4) Repeat the last two steps for the remaining images of the set.

# NOTE: compute the homographies by RANSAC (we need at least 4 corresponding points)


SRC_IMAGES_FOLDER = "images"
SRC_TEST = ["test_1", "test_2", "test_3"]

CONSIDERED = SRC_TEST[2]

DST_MOSAICING_FOLDER = "mosaicing"

# Read the images
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(f"{SRC_IMAGES_FOLDER}/{CONSIDERED}")))
images = []

# For every given path we read the corrisponding image and we put it in the list of images
for path in imagePaths:
    img = cv.imread(path)
    images.append(img)

if len(images) == 0:
    sys.exit()

# compute the salient points with SIFT for every image
sift = cv.SIFT_create()
points = []
for i, img in enumerate(images):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    gray = cv.drawKeypoints(gray, kp, gray, cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imwrite(f"gray_{i+1}.jpg", gray)
    points.append((kp, des))

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=500)
flann = cv.FlannBasedMatcher(index_params, search_params)

#compute the max_width and height of the image
# width = 0
# height = img.shape[0]
# for img in images:
#     width += img.shape[1]

ref_img = images.pop(0) # I take the first image as a reference for the matching with the others
(ref_kp, ref_des) = points.pop(0)
width = ref_img.shape[1]
height = ref_img.shape[0]
for image in images:
    width += image.shape[1]
    height += img.shape[0]

MIN_MATCH_COUNT = 4 # we want a minimum of 4 number of matching points to estimate the homography
frame = np.zeros((height, width, 3), dtype = "uint8")
frame[0:ref_img.shape[0], 0:ref_img.shape[1]] = ref_img
ref_img = frame
plt.imshow(ref_img), plt.show()
i = 0
while i < len(images):
    img = images[i]
    (kp, des) = points[i]
    matches = flann.knnMatch(des, ref_des, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) >= MIN_MATCH_COUNT:
        print(f"[INFO] match between reference and image_{i} found...")
        src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1,1,2) # source points -> where I am
        dst_pts = np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1,1,2) #destination points -> where I want to be
        # compute the homography
        H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        # width += img.shape[1]
        result = cv.warpPerspective(img, H, (width, height))
        plt.title("Warping image"), plt.imshow(result), plt.show()
        # result[0:ref_img.shape[0], 0:ref_img.shape[1]] = ref_img
        # ref_img = result
        # ref_img = cv.addWeighted(src1=ref_img, alpha=0.8, src2=result, beta=0.5, gamma=0)
        # ref_img = cv.add(src1=ref_img, src2=result)
        ref_img = np.where(result == 0, np.add(ref_img, result), result)
        plt.title("Result"),plt.imshow(ref_img), plt.savefig(f"{DST_MOSAICING_FOLDER}/{CONSIDERED}/mosaicing_{i}.jpg"), plt.show()

        (ref_kp, ref_des) = sift.detectAndCompute(ref_img, None)
        i = i + 1
    else:
        print(f"[INFO] not enough matches are found for image_{i}")



