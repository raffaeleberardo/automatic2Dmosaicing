import cv2 as cv
import sys
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import distance_transform_edt
import os
from tqdm import tqdm

def goodMatches(ref_des, des, matcher):
    '''

    :param ref_des: reference descriptors of reference image ref_img
    :param des: descriptors of image img
    :param matcher: matcher use to compute near points between ref_img and img
    :return: good found matches
    '''
    # compute the matches and keep 2 nearest neighbour
    matches = matcher.knnMatch(des, ref_des, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    return good

def createFrame(ref_img, img):
    '''

    Make a border around the ref_img equal to the shape of img. We use this as a frame where we paste the warped img.

    :param ref_img: reference image
    :param img: image with which we want to make the stitching
    :return: new ref_img
    '''
    top = int(img.shape[0])
    bottom = top
    right = int(img.shape[1])
    left = right

    ref_img = cv.copyMakeBorder(ref_img, top, bottom, left, right, cv.BORDER_CONSTANT, None, 0)
    return ref_img

def blending(I1, I2):

    '''

    :param I1: first image with which we do the blending
    :param I2: second image with which we do the blending
    :return: Blended image

     WEIGHTENED BLENDING:
     I_blended = (I1 * w1 + I2 * w2)/(w1 + w2)

     For us I1 = ref_img, I2 = warped_img
     distance_transform_edt gives more weight to the px in the centre of the image and less on the borders.
     Link: https://www.youtube.com/watch?v=D9rAOAL12SY

    '''

    w1 = distance_transform_edt(I1) # This command correspond to the bwdist() in MATLAB
    w1 = np.divide(w1, np.max(w1))
    w2 = distance_transform_edt(I2)
    w2 = np.divide(w2, np.max(w2))

    # STITCHING THE TWO IMAGES
    I_blended = np.add(np.multiply(I1, w1), np.multiply(I2, w2))
    w_tot = w1 + w2
    I_blended = np.divide(I_blended, w_tot, out=np.zeros_like(I_blended), where=w_tot != 0).astype("uint8")
    return I_blended

def cropping(img):
    '''

    :param img: image where we delete the black background
    :return: image without black background
    '''
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv.boundingRect(cnt)
    crop = img[y:y + h, x:x + w]
    return crop

def main():
    '''
    ALGORITHM : Planar panoramic mosaicing (page 206-207 of Multiple View Geometry for Computer Vision (2nd edition))
     1) Choose one image of the set as a reference
     2) Compute the homography H which maps one of the other images of the set to this reference image
     3) Projectively warp the image with this homography, and augment the reference image with the non-overlapping part of
        warped image
     4) Repeat the last two steps for the remaining images of the set.

     NOTE: compute the homographies by RANSAC (we need at least 4 corresponding points but we impose 50 to avoid matching with wrong images)
    '''

    # BEGIN SET-UP

    # folder where we have our test images
    SRC_IMAGES_FOLDER = "images"

    # folders where we keep test images for folder images and where we keep results for mosaicing folder
    SRC_TEST = ["room1",  # 0
                "bridge",  # 1
                "building_site",  # 2
                "big_house",  # 3
                "river",  # 4
                "roof",  # 5
                "not_common",  # 6
                "carmel",  # 7
                "golden_gate",  # 8
                "halfdome",  # 9
                "diamondhead",  # 10
                "fishbowl",  # 11
                "shangai",  # 12
                "yard",  # 13
                "rio",  # 14
                "office",  # 15
                "hotel",  # 16
                "alpe_siusi"]  # 17

    # selected test folder. It is the folder with the images for the stitching
    # and also it is the folder where we store the results
    CONSIDERED = SRC_TEST[17]

    # folder where we keep the results
    SRC_TEST = "source_images"
    DST_MOSAICING_FOLDER = "mosaicing"

    # folder in which we have the features points of our tested images
    DST_SIFT_FOLDER = "sift"

    SAVE_PATH = os.path.join(SRC_IMAGES_FOLDER, CONSIDERED)
    SRC_IMAGES = os.path.join(SAVE_PATH, SRC_TEST)
    SAVE_MOSAICING = os.path.join(SAVE_PATH, DST_MOSAICING_FOLDER)
    SAVE_SIFT = os.path.join(SAVE_PATH, DST_SIFT_FOLDER)

    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(SAVE_MOSAICING, exist_ok=True)
    os.makedirs(SAVE_SIFT, exist_ok=True)

    # -- END SET-UP

    # Read the images

    print("[INFO] loading images...")
    imagePaths = sorted(list(paths.list_images(f"{SRC_IMAGES}"))) # for example load all paths from: images/test_1
    images = []

    # For every given path we read the corresponding image and we put it in the list of images
    for path in imagePaths:
        img = cv.imread(path)
        images.append(img)

    # this means that we don't read images
    if len(images) == 0:
        print("[INFO] No image to read!")
        sys.exit()

    print("[INFO] Finish to read the images")

    # compute the salient points with the SIFT method for every image

    sift = cv.SIFT_create()
    points = []

    for i, img in enumerate(images):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        gray = cv.drawKeypoints(gray, kp, gray, cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imwrite(f"{SAVE_SIFT}/gray_{i+1}.jpg", gray)
        points.append((kp, des))

    print("[INFO] SIFT computed for every image in the list")

    # I take the first image as a reference for the matching with the others
    first = 0  #  (len(images)-1)//2
    ref_img = images.pop(first)
    del points[first]

    # we use this to compute the matches between pair of images
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=2)
    search_params = dict() #  dict(checks=500)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # we want a minimum of 4 number of matching points to estimate the homography
    # we want also enough points to discriminate an image from one that isn't a representation of the same scene, this is why we choose a greater number than 4,
    # 50 in our case.

    MIN_MATCH_COUNT = 10

    print("[INFO] Working on the stitching...")

    while len(images) > 0:

        max_matching = []
        best_params = {}

        print("[INFO] Searching the image with the large number of correspondencies...")

        for i in tqdm(range(len(images))):

            img = images[i]
            (kp, des) = points[i]


            ref_img = createFrame(ref_img, img)  # create a frame where we put the ref_img
            gray = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)
            (ref_kp, ref_des) = sift.detectAndCompute(gray, None)

            if len(ref_kp) >= 2:
                matches = goodMatches(ref_des=ref_des, des=des, matcher=flann)
                if len(max_matching) < len(matches):
                    max_matching = matches
                    best_params = {
                        "matches" : max_matching,
                        "ref_kp" : ref_kp,
                        "ref_des" : ref_des,
                        "idx" : i,
                        "ref_img" : ref_img
                    }
            else:
                print("[WARNING] Reference image has not enough keypoints")
                sys.exit()
            ref_img = cropping(ref_img)

        ref_img = best_params["ref_img"]
        (ref_kp, ref_des) = (best_params["ref_kp"], best_params["ref_des"])
        i = best_params["idx"]
        # (kp, des) = (best_params["kp"], best_params["des"])
        matches = best_params["matches"]
        width = ref_img.shape[1]
        height = ref_img.shape[0]

        print(f"[INFO] Number of good matches: {len(matches)}")

        if len(matches) >= MIN_MATCH_COUNT:
            img = images.pop(i)
            (kp, des) = points.pop(i)
            print(f"[INFO] Match between reference and image_{i} found...")
            src_pts = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1,1,2) # source points -> where I am
            dst_pts = np.float32([ref_kp[m.trainIdx].pt for m in matches]).reshape(-1,1,2) # destination points -> where I want to be
            # compute the homography
            H, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            print("[INFO] warping...")

            warped_img = cv.warpPerspective(img, H, (width, height))
            plt.title("Warping image"), plt.imshow(warped_img), plt.show()
            no_blend = cv.add(ref_img, warped_img)
            plt.title("Result without blending"), plt.imshow(no_blend), plt.show()
            print("[INFO] blending...")
            # If we are sure that the lighting is preserve in all the pictures, we could also use this sentence
            # ref_img = np.where(ref_img == 0, warped_img, ref_img)
            ref_img = blending(ref_img, warped_img)
            ref_img = cropping(ref_img)
            cv.imwrite(f"{SAVE_MOSAICING}/final_result.jpg", ref_img)
            print("[INFO] blending finished...")
            plt.title("Result after the blending"),plt.imshow(ref_img), plt.savefig(f"{SAVE_MOSAICING}/mosaicing_{i}.jpg"), plt.show()

        else:
            print(f"[WARNING] not enough matches found for image_{i}")
            break


if __name__ == "__main__":
    main()