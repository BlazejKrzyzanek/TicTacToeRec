from skimage import io, transform
from board import find_boards
from game import find_game_state
import glob

images = []


def import_files(directory='*', file='*.jpg'):
    for file in glob.glob("../data/" + directory + "/" + file):
        image = io.imread(file)
        image = transform.resize(image, (500, 500))
        images.append(image)


def show_images():
    for image in images:
        io.imshow(image)
        io.show()


def show_result(game_states):
    # TODO pretty image displaying
    pass


def calculate():
    for image in images:
        boards = find_boards(image)
        game_states = []
        for board in boards:
            game_states.append(find_game_state(board))

        show_result(game_states)


if __name__ == "__main__":
    import_files("easy")
    calculate()
    show_images()
