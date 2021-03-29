
import cv2 as cv
import imutils
from imutils import paths
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True,
                help="path to input directory of images to stitch")
ap.add_argument("-o", "--output", type=str, required=True,
                help="path to the output image")
ap.add_argument("-c", "--crop", type=int, default=0,
                help="wheter to crop out largest rectangular region")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["images"])))
images = []

for imagePath in imagePaths:
    image = cv.imread(imagePath)
    images.append(image)

print("[INFO] Stitching images")
stitcher = cv.createStitcher() if imutils.is_cv3() else cv.Stitcher_create()
status, stitched = stitcher.stitch(images) # I can pass directly th list of images

if status == 0:

    if args["crop"] > 0:
        print("[INFO] cropping...")
        stitched = cv.copyMakeBorder(stitched, 10, 10, 10, 10,
                                     cv.BORDER_CONSTANT, (0, 0, 0))
        gray = cv.cvtColor(stitched, cv.COLOR_BGR2GRAY)
        thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]

        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv.contourArea)

        mask = np.zeros(thresh.shape, dtype="uint8")
        (x, y, w, h) = cv.boundingRect(c)
        cv.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        minMask = mask.copy()
        sub = mask.copy()

        while cv.countNonZero(sub) > 0:
            minMask = cv.erode(minMask, None)
            sub = cv.subtract(minMask, thresh)
        cnts = cv.findContours(minMask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv.contourArea)
        (x, y, w, h) = cv.boundingRect(c)
        stitched = stitched[y:y+h, x:x+w]

    cv.imshow("Stitched", stitched)
    k = cv.waitKey(0)
    if k == ord("s"):
        cv.imwrite(args['output'], stitched)
    else:
        print(f"[ERROR] Image stitching failed with status: {status}")

