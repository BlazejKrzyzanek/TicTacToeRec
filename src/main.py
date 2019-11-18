from skimage import io, transform
from board import find_boards
from game import find_game_state
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import glob

images = []


def import_files(directory='*', file='*.jpg'):
    for file in glob.glob("../data/" + directory + "/" + file):
        image = io.imread(file)
        image = transform.resize(image, (250, 250))
        images.append([image])


def show_images():

    max_len = 0

    for i in images:
        max_len = max(max_len, len(i))

    plt.figure(figsize=(max_len * 5, len(images) * 5))

    for i, im in enumerate(images):
        for j, img in enumerate(images[i]):
            plt.subplot(len(images), max_len, i * max_len + j + 1)
            plt.axis("off")
            plt.imshow(img, cmap=cm.gray, vmin=0., vmax=1.)

    plt.savefig('boards.pdf', bbox_inches='tight')
    plt.show()


def show_result(game_states):

    pass


def calculate():
    for image in images:
        boards = find_boards(image[0])
        # image += boards
        game_states = []
        for board in boards:
            game_states += find_game_state(board)

        image += game_states


if __name__ == "__main__":
    import_files("easy")
    calculate()
    show_images()
