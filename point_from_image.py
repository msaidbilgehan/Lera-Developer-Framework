#cython: language_level=3, boundscheck=False
"""
python37 .\tools\point_from_image.py .\pattern\2021.02.09_referans

Arguments:
 1) Path of images to extract points
 2) Preprocess Flag 0-1 for Off-On

"""

# importing the module
import sys

# sys.path.append("../tools/")
import libs

from stdo import stdo

from image_manipulation import canny_edge_detection
from image_tools import (
    open_image,
    show_image,
)
from tools import (
    list_files,
    save_to_json,
)


class image_point_picker:
    last_coordinates = list()
    is_threading_active = None
    is_verbose = None

    def __init__(self, is_verbose=False, is_threading_active=False):
        self.is_threading_active = is_threading_active
        self.is_verbose = is_verbose

    def run(
        self,
        images_directory_path,
        is_preprocess_on=False,
        event_click=True,
        event_motion=True,
        necessary_events=["event_plot_dblclick"],
        is_verbose=None,
    ):
        sources_list = list()

        if is_verbose is None:
            is_verbose = self.is_verbose

        if images_directory_path[-1] == "\\" or images_directory_path[-1] == "/":
            images_directory_path = images_directory_path[:-1]

        for images_path in list_files(images_directory_path):
            current_image = open_image(images_path, option="cv2-rgb", is_numpy=True)
            if is_preprocess_on:
                current_image = image_point_picker.preprocess_sequence(current_image)
            sources_list.append(current_image)

        if is_preprocess_on:
            cmap = "gray"
        else:
            cmap = None

        if len(sources_list) > 1:
            open_order = 2
        else:
            open_order = 1

        click_coordinates = show_image(
            sources_list,
            title="Image Point Picker",
            option="plot",
            cmap=cmap,
            open_order=open_order,
            window=True,
            event_click=event_click,
            event_motion=event_motion,
            is_verbose=is_verbose,
        )
        stdo(1, "Data of click_coordinates:\n\t{}".format(click_coordinates))

        json_data = dict()
        for key, value in click_coordinates.items():
            if key in necessary_events:
                json_data[key] = value

        stdo(1, "Json Data is {}".format(json_data))
        save_to_json(json_data, "coordinates.json", dict_format=True)

    @staticmethod
    def preprocess_sequence(image):
        configs = []
        # configs = [0, 0]
        # cany_edge = canny_edge_detection(image, configs)
        configs = []
        preprocessed_image = canny_edge_detection(image, configs)

        return preprocessed_image


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
    images_directory_path = "images/"
    is_preprocess_on = 0

    if len(sys.argv) == 3:
        images_directory_path = sys.argv[1]
        is_preprocess_on = int(sys.argv[2])
    elif len(sys.argv) == 2:
        images_directory_path = sys.argv[1]

    """
    # Events List
    necessary_events = [
        "event_plot_click",
        "event_nonplot_click",
        "event_plot_dblclick",
        "event_nonplot_dblclick",
        "event_plot_motion",
        "event_nonplot_motion",
    ]
    """

    necessary_events = ["event_plot_dblclick"]

    ipp = image_point_picker()
    ipp.run(
        images_directory_path,
        is_preprocess_on=is_preprocess_on,
        event_click=True,
        event_motion=False,
        necessary_events=necessary_events,
        is_verbose=True,
    )
