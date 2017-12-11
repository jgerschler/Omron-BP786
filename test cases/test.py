#!/usr/bin/env python

"""
Proof of concept script; As of this writing, the Omron BP786 does not currently
support data export outside of a closed-source app. Code modified from
http://www.pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/
"""

from imutils import contours
import imutils
import cv2
import time
 
class OmronBP786(object):
    """
    A class for reading the display of an Omron BP786 blood pressure monitor.
    """
    DIGITS_LOOKUP = {
        (1, 1, 1, 0, 1, 1, 1): 0,
        (0, 0, 1, 0, 0, 1, 0): 1,
        (1, 0, 1, 1, 1, 1, 0): 2,
        (1, 0, 1, 1, 0, 1, 1): 3,
        (0, 1, 1, 1, 0, 1, 0): 4,
        (1, 1, 0, 1, 0, 1, 1): 5,
        (1, 1, 0, 1, 1, 1, 1): 6,
        (1, 1, 1, 0, 0, 1, 0): 7,
        (1, 1, 1, 1, 1, 1, 1): 8,
        (1, 1, 1, 1, 0, 1, 1): 9
    }

    def __init__(self):
        image_file = 'sample.png'

    def acquire(self, image_file='sample.png'):        
        self.image = imutils.resize(cv2.imread(image_file), height=500)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (39, 39), 0)# kernel width and height should be odd
        thresh = cv2.threshold(blurred, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
        self.thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        digit_coords = []
         
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if float(h) / w > 1.1 and h > 70:# only grab large countours
                digit_coords.append((x, y, w, h))
                
        digit_coords.remove(max(digit_coords))# remove right-most artefact (not a digit)

        digit_coords.sort(key=lambda x:x[1])

        self.bin_list = [[digit_coords.pop(0)]]

        for rect in digit_coords:
            if abs(rect[1] - self.bin_list[0][0][1]) < 20:
                self.bin_list[0].append(rect)
                digit_coords = digit_coords[1:]

        self.bin_list.append([digit_coords.pop(0)])

        for rect in digit_coords:
            if abs(rect[1] - self.bin_list[1][0][1]) < 20:
                self.bin_list[1].append(rect)
                digit_coords = digit_coords[1:]

        self.bin_list.append(digit_coords)

        for entry in self.bin_list:
            entry.sort()

        for digit_coords in self.bin_list[0]:
            self.digits = []
            self.compute_segments(digit_coords, 0.25, 0.15, 0.15)
            
    def compute_segments(self, digit_coords, dW_factor, dH_factor, dHC_factor):
        (x, y, w, h) = digit_coords
        roi = self.thresh[y:y + h, x:x + w]
        (roiH, roiW) = roi.shape

        (dW, dH) = (int(roiW * dW_factor), int(roiH * dH_factor))
        dHC = int(roiH * dHC_factor)

        segments = [
            ((0, 0), (w, dH)),  # top
            ((0, 0), (dW, h // 2)), # top-left
            ((w - dW, 0), (w, h // 2)), # top-right
            ((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
            ((0, h // 2), (dW, h)), # bottom-left
            ((w - dW, h // 2), (w, h)), # bottom-right
            ((0, h - dH), (w, h))   # bottom
        ]

        on = [0] * len(segments)

        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
            segROI = roi[yA:yB, xA:xB]
            total = cv2.countNonZero(segROI)
            area = (xB - xA) * (yB - yA)

            if total / float(area) > 0.5:# greater than 50% of pixels are non-zero, mark segment as on
                on[i]= 1

        # lookup the digit and draw it on the image
        digit = OmronBP786.DIGITS_LOOKUP[tuple(on)]
        self.digits.append(str(digit))

        cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 0, 255), 1)


if __name__ == '__main__':
    bp = OmronBP786()
    bp.acquire()# acquire data from image
    cv2.imshow("Output", bp.image)# show image
    cv2.waitKey(0)
