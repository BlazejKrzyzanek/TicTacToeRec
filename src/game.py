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

    # linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 20, None, 35, 3)
    #
    # if linesP is not None:
    #     for x in range(0, len(linesP)):
    #         l = linesP[x][0]
    #         cv.line(cdstP, (l[0], l[1]), (l[2], l[3]),
    #                 (0, np.random.randint(127, 255), np.random.randint(127, 255)), 2,
    #                 cv.LINE_AA)

    img = ski.img_as_float(img)
    src = ski.img_as_float(src)
    cdstP = ski.img_as_float(cdstP)

    squares = ski.img_as_bool(dst)
    squares = mp.remove_small_holes(squares, 120)
    squares = mp.dilation(squares, mp.disk(1))
    squares = mp.thin(squares)
    squares = ndi.binary_fill_holes(squares)
    squares = mp.erosion(squares, mp.disk(3))
    squares = mp.remove_small_objects(squares, 45)
    squares = ski.img_as_ubyte(squares)

    contours, hierarchy = cv.findContours(squares, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for c in contours:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.05 * peri, True)

        x0, x1, x2, x3, x4, x5 = 15, 35, 37, 57, 60, 80
        if len(approx) == 4:
            fields = [dst[x0:x1, x0:x1], dst[x0:x1, x2:x3], dst[x0:x1, x4:x5],
                      dst[x2:x3, x0:x1], dst[x2:x3, x2:x3], dst[x2:x3, x4:x5],
                      dst[x4:x5, x0:x1], dst[x4:x5, x2:x3], dst[x4:x5, x4:x5]]

            for i, field in enumerate(fields):
                field = ski.img_as_ubyte(field)
                contours, hierarchy = cv.findContours(field, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

                fields[i] = cv.cvtColor(field, cv.COLOR_GRAY2BGR)

                # Remove edges
                for c in contours:
                    peri = cv.arcLength(c, True)
                    approx = cv.approxPolyDP(c, 0.02 * peri, True)

                    if len(approx) <= 4:
                        cv.fillPoly(fields[i], [c], (0, 0, 0))

                m = np.mean(field[5:-5, 5:-5]) / 255
                if m < 0.1:
                    state.append(' ')
                    continue

                field_filled = ndi.binary_fill_holes(field)
                field_filled = ski.img_as_ubyte(field_filled)
                contours, hierarchy = cv.findContours(field_filled, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

                circle = False

                for c in contours:
                    peri = cv.arcLength(c, True)
                    approx = cv.approxPolyDP(c, 0.05 * peri, True)

                    area = cv.contourArea(c)
                    perimeter = cv.arcLength(c, False)
                    if perimeter == 0.0:
                        continue

                    sqrt_area = np.sqrt(area)
                    ratio = float(sqrt_area) / perimeter

                    if ratio > 0.22:
                        cv.drawContours(fields[i], [c], -1, (255, 0, 0), 1)
                        circle = True
                        break
                    else:
                        cv.drawContours(fields[i], [c], -1, (255, 0, 255), 1)

                if circle:
                    state.append('o')
                else:
                    state.append('x')

    if len(state) != 0:
        state = [state[:3], state[3:6], state[6:]]
        return [cdstP], state
    else:
        return [cdstP], [['', '', ''], ['', '', ''], ['', '', '']]
