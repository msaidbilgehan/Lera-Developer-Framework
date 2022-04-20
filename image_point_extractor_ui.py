#cython: language_level=3, boundscheck=False
# importing the module
import sys

# sys.path.append("../tools/")
import libs

from image_tools import (
    open_image,
    show_image,
)
from tools import (
    list_files,
    save_to_json,
    list_to_dict,
)


class image_point_picker:
    last_coordinates = list()
    is_threading_active = None
    is_verbose = None

    def __init__(self, is_verbose=False, is_threading_active=False):
        self.is_threading_active = is_threading_active
        self.is_verbose = is_verbose

    def run(self, images_directory_path, is_preprocess_on=False):
        sources_list = list()

        for images_path in list_files(images_directory_path):
            sources_list.append(
                open_image(images_path, option="cv2-rgb", is_numpy=True)
            )

        click_coordinates = show_image(
            sources_list,
            title="Image Point Picker",
            option="plot",
            cmap=None,  # "gray",
            open_order=2,
            window=True,
            event_click=True,
        )
        save_to_json(list_to_dict(click_coordinates), "coordinates.json")


"""
        image = cv2.imread(image_one_path, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image2 = cv2.imread(image_two_path, 1)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        fig = plt.figure()
        a1 = fig.add_subplot(1, 2, 1)
        a2 = fig.add_subplot(1, 2, 2)

        a1.imshow(image)
        a2.imshow(image2)
        coords = []
        coords = fig.canvas.mpl_connect("button_press_event", onclick)
        plt.show()

        print("coords: ", coords)

    def onclick(self, event, is_verbose=True):
        if event.dblclick:
            ix, iy = event.xdata, event.ydata

            if self.is_verbose:
                print("Choosen: x = %d, y = %d" % (ix, iy))

            self.coords.append((round(ix), round(iy)))
"""


# driver function
if __name__ == "__main__":
    ipp = image_point_picker()
    ipp.run(sys.argv[1], is_preprocess_on=True)
