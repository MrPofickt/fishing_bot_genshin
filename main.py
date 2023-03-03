import numpy as np
import cv2
import time

import pyautogui
import pyscreenshot as pss
import mss
import pytesseract
from ahk import AHK

# Set path to tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize AHK object
ahk = AHK()

if __name__ == '__main__':
    last_time = time.time()
    title = "Auto-Fishing"
    sct = mss.mss()
    mon = {"top": 165, "left": 730, "width": 400, "height": 180}
    start = False

    while True:
        x_past = 0
        y_past = 0
        # Only process image every 2 seconds
        if time.time() - last_time < 2:
            continue

        sct_img = sct.grab(mon)

        # Convert to numpy array
        img = np.array(sct_img)

        # Get image shape
        h, w, _ = img.shape

        # Convert image to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define color range for yellow
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        # Create mask for yellow pixels
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Apply mask to original image to get yellow pixels only
        yellow_img = cv2.bitwise_and(img, img, mask=mask)

        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Display images
        cv2.imshow(title, gray)
        cv2.imshow('Yellow Image', yellow_img)

        # Apply thresholding to grayscale image
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # Save thresholded image to file
        cv2.imwrite('temp.jpg', thresh)

        # Read image from file and apply OCR
        img = cv2.imread('temp.jpg')
        text = pytesseract.image_to_string(img)
        print(text)

        # Check if text contains certain string
        if "You've got a bite!" in text:
            print("+")
            ahk.click()
            start = True

        if start:
            # Find contours on the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])

                # Create array of coordinates
                loc = [(center_x, center_y)]

                for pt in loc:
                    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)
                    x = pt[0]
                    y = pt[1]
                    print(x)
                    print(y)
                    if 0 < x < 100:
                        pyautogui.mouseDown(button='left')
                        time.sleep(0.5)
                        pyautogui.mouseUp(button='left')
                        x = 0
                        break
                    if 100 < x < 200:
                        pyautogui.mouseDown(button='left')
                        time.sleep(1)
                        pyautogui.mouseUp(button='left')
                        x = 0
                        break
                    if 200 < x < 300:
                        pyautogui.mouseDown(button='left')
                        time.sleep(1.5)
                        pyautogui.mouseUp(button='left')
                        x = 0
                        break
                    if 300 < x < 400:
                        pyautogui.mouseDown(button='left')
                        time.sleep(2)
                        pyautogui.mouseUp(button='left')
                        x = 0
                        break
                    if 400 < x < 500:
                        pyautogui.mouseDown(button='left')
                        time.sleep(2.5)
                        pyautogui.mouseUp(button='left')
                        x = 0
                        break
                    if 500 < x < 600:
                        pyautogui.mouseDown(button='left')
                        time.sleep(3)
                        pyautogui.mouseUp(button='left')
                        x = 0
                        break
                    if 600 < x < 700:
                        pyautogui.mouseDown(button='left')
                        time.sleep(3.5)
                        pyautogui.mouseUp(button='left')
                        x = 0
                        break

                        # Wait for a moment before capturing next image
        time.sleep(0.1)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            quit()