import cv2
import numpy as np
import skimage as ski
import skimage.morphology as mp
from skimage.filters.rank import mean_bilateral
from scipy import ndimage as ndi
from skimage import io


def show_boards(boards):
    for board in boards:
        io.imshow(board)
        io.show()


def transform_src_pts(src_pts):
    A = src_pts[0]
    B = src_pts[1]
    C = src_pts[2]
    D = src_pts[3]

    AD = [(A[0] - D[0]) * 2, (A[1] - D[1]) * 2]
    BD = [(B[0] - D[0]) * 2, (B[1] - D[1]) * 2]
    CD = [(C[0] - D[0]) * 2, (C[1] - D[1]) * 2]
    A += AD
    B += BD
    C += CD

    return np.array([A, B, C, D])


def find_boards(img):
    boards = []
    debug_img = ski.img_as_ubyte(img)

    img = ski.img_as_ubyte(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 40, 70)

    boards_img = ski.img_as_bool(edges)
    boards_img = mp.remove_small_holes(boards_img, 250)
    boards_img = mp.dilation(boards_img, mp.disk(1))
    boards_img = mp.thin(boards_img)
    boards_img = ndi.binary_fill_holes(boards_img)

    squares = mp.erosion(boards_img, mp.disk(3))
    squares = mp.remove_small_objects(squares, 45)
    squares = ski.img_as_ubyte(squares)

    contours, hierarchy = cv2.findContours(squares, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    potential_boards = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)

        if len(approx) == 4:
            potential_boards.append(approx)
            # cv2.drawContours(img, approx, -1, (0, 0, 255), 5)

    for b in potential_boards:
        M = cv2.moments(b)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        rotated_rect = cv2.minAreaRect(b)
        (x, y), (w, h), angle = rotated_rect

        box = cv2.boxPoints(rotated_rect)
        box = np.int0(box)

        src_pts = box.astype("float32")
        src_pts = transform_src_pts(src_pts)

        dst_pts = np.array([[0, h],
                            [0, 0],
                            [w, 0],
                            [w, h]], dtype="float32")

        dst_pts *= 3

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        bg_color = np.mean(img)

        warped = cv2.warpPerspective(img, M, (int(w * 1.8 * 3), int(h * 1.8 * 3)), borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(bg_color, bg_color, bg_color))

        gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
        gray = mean_bilateral(ski.img_as_ubyte(gray), mp.disk(5), s0=10, s1=10)
        gray = ski.img_as_float(gray)
        gray = 1 - gray

        gray = cv2.resize(gray, (100, 100))
        boards.append(ski.img_as_float(gray))

    cv2.destroyAllWindows()

    # show_boards(boards)
    return boards
