import cv2
import PIL.Image
import pytesseract
from pytesseract import Output
import numpy as np


def preprocessImage(file_name):
    """
        NOTE: Saw better text detection using this, previously undetected text was now picke up
        :param file_name:
        :return:
    """
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = adjustLighting(img)
    img = sharpenImage(img)

    return img


def sharpenImage(img):
    """
    sharpens the image, reduce blur
    these transformations make it easier for the ocr algorithm to detect text
    :param img:
    :return:
    """

    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv2.filter2D(img, -1, sharpen_kernel)
    return sharpen


def adjustLighting(img):
    """
    This function aims to adjust the various lighting discrepancies in an image to make it more legible
    The block size determines how many pixels are used to determine the value for the GAUSSIAN algorithm
    and the parameter c is used for noise reduction

    The GAUSSIAN option uses a gaussian window, to get weighted sum of the individual pixels when applying the changes

    :param file:
    :return:
    """
    BLOCK_SIZE = 41

    # _, res = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY) # non adaptive
    res = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=BLOCK_SIZE, C=4)

    return res


def detectTextOCR(file_name):
    # oem - 3 is default engine mode, psm -11
    """
    Page segmentation modes:
      0    Orientation and script detection (OSD) only.
      1    Automatic page segmentation with OSD.
      2    Automatic page segmentation, but no OSD, or OCR.
      3    Fully automatic page segmentation, but no OSD. (Default)
      4    Assume a single column of text of variable sizes.
      5    Assume a single uniform block of vertically aligned text.
      6    Assume a single uniform block of text.
      7    Treat the image as a single text line.
      8    Treat the image as a single word.
      9    Treat the image as a single word in a circle.
     10    Treat the image as a single character.
     11    Sparse text. Find as much text as possible in no particular order.
     12    Sparse text with OSD.
     13    Raw line. Treat the image as a single text line,
           bypassing hacks that are Tesseract-specific.
    """

    TESSERACT_CONFIG = r"--psm 11 --oem 3"

    # Read Image and convert to grayscale
    # Converting to grayscale to eliminate additional computational complexities

    img_processed = preprocessImage(file_name)
    img_output = cv2.imread(file_name)
    # Get image dimensions
    img_height, img_width = img_processed.shape

    # Extract Data from `the image using the OCR algorithm and return the output as a dictionary object
    # Pass in Config values for Tesseract
    img_data = pytesseract.image_to_data(img_processed, config=TESSERACT_CONFIG, output_type=Output.DICT)

    textObjsDetectedCount = len(img_data["text"])

    # Iterate over all of the detected text objects
    for i in range(textObjsDetectedCount):
        # Filter based on confidence value of the detected text
        if float(img_data["conf"][i] >= 20):
            x = img_data["left"][i]
            y = img_data["top"][i]
            width = img_data["width"][i]
            height = img_data["height"][i]
            curr_text = img_data["text"][i]

            img_output = cv2.rectangle(img_output, (x,y), (width+x, height+y), (0, 255, 0), 2)
            img_output = cv2.putText(img_output, curr_text, (x, height+y), cv2.FONT_HERSHEY_PLAIN, 1, (0,0, 255))

    print(pytesseract.image_to_string(img_processed, config=TESSERACT_CONFIG))
    cv2.imshow("Processed Image", img_processed)
    cv2.imshow("Text Detection", img_output)
    cv2.waitKey(0)


def main():
    file_path = "test6.png"
    detectTextOCR(file_path)



if __name__ == "__main__":
    print("Press 0 to exit \n ----------------------------------- \n")
    main()