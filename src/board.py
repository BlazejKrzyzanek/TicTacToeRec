import cv2
import numpy as np
import skimage as ski
import skimage.morphology as mp
from skimage import io


def show_boards(boards):
    for board in boards:
        io.imshow(board)
        io.show()


def get_angle_to_rotate(contour):
    rect = cv2.minAreaRect(contour)
    angle = rect[2]

    if angle < -45:
        angle = (90 + angle)

    return angle
    a = contour[0]
    b = contour[1]

    x0, y0 = a[0][0], a[0][1]
    x1, y1 = b[0][0], b[0][1]
    delta_y = y0 - y1
    delta_x = x0 - x1

    return np.tan(delta_y / delta_x)


def crop_board(img, cx, cy, w, h):
    img_w, img_h = img.shape[0], img.shape[1]

    x0 = max(int(cx - 4 * w), 0)
    x2 = min(int(cx + 4 * w), img_w)
    y0 = max(int(cy - 4 * h), 0)
    y2 = min(int(cy + 4 * h), img_h)

    return img[y0:y2, x0:x2]


def rotate_img(image, angle, center=None):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def debug_show(img):
    boards_img = ski.img_as_ubyte(img)
    cv2.imshow("Image", boards_img)
    k = cv2.waitKey(0)


def find_boards(img):
    boards = []

    img = ski.img_as_ubyte(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 70)

    # boards_img = mp.dilation(edges, mp.disk(1))
    # boards_img = ndi.binary_fill_holes(boards_img)
    # boards_img = mp.erosion(boards_img, mp.disk(1))
    boards_img = mp.remove_small_holes(edges, 250)
    debug_show(boards_img)
    boards_img = mp.dilation(edges, mp.disk(1))
    boards_img = mp.thin(boards_img)
    # boards_img = ndi.binary_fill_holes(boards_img)
    # boards_img = mp.thin(boards_img)

    squares = mp.erosion(boards_img, mp.disk(8))
    squares = mp.opening(squares, mp.disk(3))
    squares = mp.remove_small_objects(squares, 45)

    squares = ski.img_as_ubyte(squares)

    contours, hierarchy = cv2.findContours(squares, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    potential_boards = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        if len(approx) == 4:
            potential_boards.append(approx)
            cv2.drawContours(img, [approx], -1, (255, 0, 0), 2)

    cropped_potential_boards = []
    for b in potential_boards:
        M = cv2.moments(b)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        (x, y), (w, h), angle = cv2.minAreaRect(b)

        rotated_img = rotate_img(img, angle, (cx, cy))
        cropped_img = crop_board(rotated_img, x, y, w, h)

        cropped_potential_boards.append(cropped_img)

    edges = cv2.Canny(squares, 50, 200, None, 3)

    boards_img = ski.img_as_ubyte(boards_img)
    debug_show(boards_img)
    lines = cv2.HoughLinesP(boards_img, 1, (np.pi / 180), 35, None, 5, 15)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
    #         pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
    #         cv2.line(img, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

    debug_show(img)

    # io.imshow(edges)
    # io.show()

    # for i in cropped_potential_boards:
    #     cv2.imshow("Image", i)
    #     k = cv2.waitKey(0)
    # cv2.imshow("Image", boards_img)
    # k = cv2.waitKey(0)
    cv2.destroyAllWindows()

    show_boards(boards)
    return boards
