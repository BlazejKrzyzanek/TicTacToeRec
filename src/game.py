import cv2 as cv
import numpy as np
import skimage as ski
from skimage import morphology as mp
from skimage.filters.rank import mean_bilateral
from scipy import ndimage as ndi



def s_curve(x):
    return 0.5 + (3 * (x - 0.5)) / 2 * np.sqrt(1 + 9 * (x - 0.5) ** 2)


def find_game_state(img):
    state = []
    # TODO should return game state as simple list - example output :
    #  [['x', '', 'o'], ['', 'x', 'x'], ['o', '', 'o']]
    #  assuming that 'board' parameter is square image with only one game board
    # return state

    src = s_curve(img)
    dst = src > ski.filters.threshold_otsu(src)

    dst = ski.img_as_ubyte(dst)

    cdstP = np.copy(cv.cvtColor(dst, cv.COLOR_GRAY2BGR))

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 20, None, 35, 3)

    if linesP is not None:
        for x in range(0, len(linesP)):
            l = linesP[x][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]),
                    (0, np.random.randint(127, 255), np.random.randint(127, 255)), 2,
                    cv.LINE_AA)

    img = ski.img_as_float(img)
    src = ski.img_as_float(src)
    cdstP = ski.img_as_float(cdstP)

    # squares = ndi.binary_fill_holes(dst)
    # squares = mp.erosion(squares, mp.disk(3))
    # squares = mp.remove_small_objects(squares, 45)
    # squares = ski.img_as_ubyte(squares)
    #
    # contours, hierarchy = cv.findContours(squares, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #
    # for c in contours:
    #     peri = cv.arcLength(c, True)
    #     approx = cv.approxPolyDP(c, 0.05 * peri, True)
    #
    #     if len(approx) == 4:
    #         cv.drawContours(dst, approx, -1, (255, 0, 0), 5)

    return [src, dst, cdstP]
