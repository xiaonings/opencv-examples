import cv2
import numpy as np
import os


def image_preprocess(gray_image):
    ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
    v = np.mean(gray_image)
    sigma = 0.33
    lower = (int(max(0, (1.0 - sigma) * v)))
    upper = (int(min(0, (1.0 + sigma) * v)))
    edged = cv2.Canny(opening, lower, upper)

    cv2.imshow('gray', gray_image)
    cv2.imshow('threshold', thresh)
    cv2.imshow('opening', opening)
    cv2.imshow('Canny', edged)
    return edged


def image_houghlinesp_detect(source):
    result = cv2.HoughLinesP(source, 1, np.pi / 180, 30, minLineLength=50, maxLineGap=10)
    lines = result[:, 0, :][:]

    return lines


def main(file):
    image = cv2.imread(file)
    image = cv2.resize(image, (800, 500), interpolation=cv2.INTER_AREA)
    cv2.imshow('original image', image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = image_preprocess(gray)
    all_lines = image_houghlinesp_detect(canny)
    for x1, y1, x2, y2 in all_lines:
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        print('x1=%s, y1=%s, x2=%s, y2=%s' % (x1, y1, x2, y2))
    cv2.imshow('result', image)


if __name__ == '__main__':
    file = os.path.join('..', 'data', 'houghlines_detect.jpg')
    main(file)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
