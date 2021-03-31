import cv2 as cv
import imutils
import sys
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import distance_transform_edt

def blending(ref_img, warped_img):
    w_ref = distance_transform_edt(ref_img)
    w_ref = np.divide(w_ref, np.max(w_ref))
    w_warp = distance_transform_edt(warped_img)
    w_warp = np.divide(w_warp, np.max(w_warp))
    ref_img = np.add(np.multiply(ref_img, w_ref), np.multiply(warped_img, w_warp))
    w_tot = w_warp + w_ref

    ref_img = np.divide(ref_img, w_tot, out=np.zeros_like(ref_img), where=w_tot != 0).astype("uint8")
    return ref_img


# ALGORITHM
# 1) Choose one image of the set as a reference
# 2) Compute the homography H which maps one of the other images of the set to this reference image
# 3) Projectively warp the image with this homography, and augment the reference image with the non-overlapping part of
#    warped image
# 4) Repeat the last two steps for the remaining images of the set.

# NOTE: compute the homographies by RANSAC (we need at least 4 corresponding points)


SRC_IMAGES_FOLDER = "images"
SRC_TEST = ["test_1", "test_2", "test_3"]

CONSIDERED = SRC_TEST[0]

DST_MOSAICING_FOLDER = "mosaicing"
DST_SIFT_FOLDER = "sift"

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
fig, ax = plt.subplots(nrows=len(images))
fig.suptitle("SIFT for each given image", fontsize=14)
for i, img in enumerate(images):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    gray = cv.drawKeypoints(gray, kp, gray, cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    ax[i].imshow(gray)
    cv.imwrite(f"{DST_MOSAICING_FOLDER}/{CONSIDERED}/{DST_SIFT_FOLDER}/gray_{i+1}.jpg", gray)
    points.append((kp, des))


plt.savefig(f"{DST_MOSAICING_FOLDER}/{CONSIDERED}/{DST_SIFT_FOLDER}/corresponding_points.jpg")
plt.show()

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=500)
flann = cv.FlannBasedMatcher(index_params, search_params)

ref_img = images.pop(0) # I take the first image as a reference for the matching with the others
(ref_kp, ref_des) = points.pop(0)
width = ref_img.shape[1]
height = ref_img.shape[0]
for image in images:
    width += image.shape[1]
    height += img.shape[0]

# we want a minimum of 4 number of matching points to estimate the homography
MIN_MATCH_COUNT = 4
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
        print("[INFO] warping...")
        warped_img = cv.warpPerspective(img, H, (width, height))
        plt.title("Warping image"), plt.imshow(warped_img), plt.show()
        print("[INFO] blending...")
        ref_img = blending(ref_img, warped_img)
        print("[INFO] blending finished...")
        # ref_img = np.where(ref_img == 0, warped_img, ref_img)
        plt.title("Result"),plt.imshow(ref_img), plt.savefig(f"{DST_MOSAICING_FOLDER}/{CONSIDERED}/mosaicing_{i}.jpg"), plt.show()

        (ref_kp, ref_des) = sift.detectAndCompute(ref_img, None)
        i = i + 1
    else:
        print(f"[INFO] not enough matches are found for image_{i}")

cv.imwrite(f"{DST_MOSAICING_FOLDER}/{CONSIDERED}/final_result.jpg", ref_img)

