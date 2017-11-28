import cv2
import numpy as np
from utils.utils import get_four_points
import os

'''
按照比例缩小图片
'''
def make_scaled_ims(im, ratio):
    shape = (int(im.shape[0]*ratio), int(im.shape[1]*ratio))
    return cv2.resize(im, (shape[1], shape[0]))


def main(image):
    size = (80, 100, 3)
    im_dst = np.zeros(size, np.uint8)
    pts_dst = np.array(
        [
            [0, 0],
            [size[0] - 1, 0],
            [size[0] - 1, size[1] - 1],
            [0, size[1] - 1]
        ], dtype=float
    )
    print('''
        Click on the four corners of the book -- top left first and
        bottom left last -- and then hit ENTER
        ''')

    img = make_scaled_ims(image, 1)
    pts_src = get_four_points(img)
    h, status = cv2.findHomography(pts_src, pts_dst)
    im_dst = cv2.warpPerspective(image, h, size[0:2])
    cv2.destroyAllWindows()
    cv2.imshow("Image", im_dst)
    cv2.waitKey(0)


if __name__ == '__main__':
    image = cv2.imread(os.path.join('..', 'data', 'perspective_corrective.jpg'))
    main(image)
