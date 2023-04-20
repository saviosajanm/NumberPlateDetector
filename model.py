from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2
from imutils import paths
import glob
from paddleocr import PaddleOCR,draw_ocr
import pytesseract
ocr = PaddleOCR(use_angle_cls=True, lang='en')


class PyImageSearchANPR:

    def locate_license_plate_candidates(self, img):
        cv2.imshow("1", img)
        #print("Original image\n\n")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("2", gray)
        #print("Grayscale of the image\n\n")

        # structuring element is rectangle
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        # performing blackhat operation on greyscale image using above kernel
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
        #cv2.imshow("3", blackhat)
        #print("performing blackhat operation on greyscale image using above kernel\n\n")

        # structuring element is square
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # performing closing operation on greyscale image using above kernel
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
        # thresholding above image using otsu method
        light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #cv2.imshow("4", light)
        #print("thresholding above image using otsu method to get the light image\n\n")

        # using sobel method to get X gradient of image
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
        gradX = np.absolute(gradX)
        # scaling resulting intensities back to the range 0-255
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        gradX = gradX.astype("uint8")
        #cv2.imshow("5", gradX)
        #print("using sobel method to get X gradient of image\n\n")

        # cleaning the x gradient by blurring, closing operation, thresholdng, erosion and dilation
        gradX = cv2.GaussianBlur(gradX, (1, 1), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #cv2.imshow("6", thresh)
        #print("Performing gaussian blur on the thesholded image\n\n")

        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        #cv2.imshow("7", thresh)
        #print("cleaning the image and reducing noise by erosion and dilation\n\n")

        # masking the light image over the thresholded image using bitwise-AND
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        # cleaning the new masked image
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        #cv2.imshow("8", thresh)
        #print("masking the light image over the thresholded image using bitwise-AND\n\n")

        # finding the contrours from the new image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # sorting the contrours in descending order by contour area
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # returning the contours
        return cnts

    def locate_license_plate(self, gray, candidates):
        lpCnt = None
        roi = None

        for c in candidates:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)

            if ar >= 4 and ar <= 5:
                lpCnt = c
                licensePlate = gray[y:y + h, x:x + w]
                roi = cv2.threshold(licensePlate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                break

        return (roi, lpCnt, (x, y, w, h))

    def find_and_ocr(self, image):
        lpText = None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidates = self.locate_license_plate_candidates(image)
        (lp, lpCnt, (x, y, w, h)) = self.locate_license_plate(gray, candidates)

        if lp is not None:
            lpText = ocr.ocr(lp, cls=True)
            cv2.imshow("9", lp)

        return (lpText, lpCnt, (x, y, w, h))


    def process(image):
        #anpr = PyImageSearchANPR()
        c = 0
        #for imagePath in glob.glob("imgs/*.png"):
        # print(imagePath)
        image = cv2.imread(image)
        image = imutils.resize(image, width=600)

        x = PyImageSearchANPR()

        (lpText, lpCnt, (x, y, w, h)) = x.find_and_ocr(image)
        #lpText = lpText + ", " + str(pytesseract.image_to_string(image, lang="eng"))
        #print(pytesseract.image_to_string(image, lang="eng"))

        files = []
        # print(imagePath)
        if (lpText == None or lpText == [[]]):
            # print("---------------------------", "No Number Plate Found", "---------------------------\n")
            return "No number plate found!"
        else:
            cv2.putText(image, lpText[0][0][1][0], (x, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            box = cv2.boxPoints(cv2.minAreaRect(lpCnt))
            box = box.astype("int")
            cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
            #cv2.imshow("final", image)
            #print("===========================", lpText[0][0][1][0], "===========================\n")
            # files.append(imagePath)
            return [lpText[0][0][1][0], pytesseract.image_to_string(image, lang="eng")]
