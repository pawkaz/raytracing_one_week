import cv2 as cv
from pathlib import Path
def main():
    cv.imwrite('test.jpg',cv.imread('test.ppm'))


if __name__ == "__main__":
    main()