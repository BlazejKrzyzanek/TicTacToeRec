import cv2
import numpy as np
from skimage import io, transform
from board import find_boards
from game import find_game_state
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import glob

original_images = []
images = []
res_i = 0


def import_files(directory='*', file='*.jpg'):
    for file in glob.glob("../data/" + directory + "/" + file):
        image = io.imread(file)
        image = transform.resize(image, (250, 250))
        original_images.append([image, file])

        img = transform.resize(image, (250, 250))
        images.append([img])


def show_images():
    max_len = 0

    for i in images:
        max_len = max(max_len, len(i))

    plt.figure(figsize=(max_len * 5, len(images) * 5))

    for i, im in enumerate(images):
        for j, img in enumerate(images[i]):
            plt.subplot(len(images), max_len, i * max_len + j + 1)
            plt.axis("off")
            # plt.imshow(img, cmap=cm.gray, vmin=0., vmax=1.)
            plt.imshow(img, cmap=cm.gray, vmin=0., vmax=1.)

    plt.savefig('boards.pdf', bbox_inches='tight')
    # plt.show()


def show_result(state):
    image = np.ones((250, 250, 3), np.uint8) * 255

    y0, dy = 50, 25
    text = ''
    text += state[0][0] + '|' + state[0][1] + '|' + state[0][2] + "\n"
    text += "____\n"
    text += state[1][0] + '|' + state[1][1] + '|' + state[1][2] + "\n"
    text += "____\n"
    text += state[2][0] + '|' + state[2][1] + '|' + state[2][2] + "\n"
    for j, line in enumerate(text.split('\n')):
        y = y0 + j * dy
        image = cv2.putText(image, line, (50, y), cv2.FONT_HERSHEY_PLAIN, 2, 2)

    return image


def calculate():
    for image, original in zip(images, original_images):
        boards = find_boards(image[0])
        # image += boards
        debug_images = []
        states = []
        for board in boards:
                debug_image, state = find_game_state(board)
                debug_images += debug_image
                debug_images += [show_result(state)]
                states.append(state)

        image += debug_images


if __name__ == "__main__":
    import_files("easy")
    calculate()
    show_images()
