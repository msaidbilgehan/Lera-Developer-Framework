#cython: language_level=3, boundscheck=False
"""
python37 .\tools\auto_image_alignment.py .\pattern\2021.02.09_referans\empty_original.png .\pattern\2021.02.09_referans\empty.png .\pattern\2021.02.09_referans\coordinates.json
"""

from __future__ import print_function
import cv2
import numpy as np
import sys
import json
from tools import load_from_json, dict_to_list, list_files
from image_tools import show_image

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

"""
def convert_list_to_tuple(list):
    return tuple(i for i in list)


def list_files(path="", name="*", extensions=["png"], recursive=False, verbose=True):
    from glob import glob

    # https://mkyong.com/python/python-how-to-list-all-files-in-a-directory/
    try:
        files = list()
        if recursive:
            if path[-1] != "/":
                path = path + "/"
        else:
            if path[-1] == "/":
                path = path[:-1]

        for extension in extensions:
            files.extend(
                [
                    f
                    for f in glob(
                        path + "**/{}.{}".format(name, extension), recursive=recursive
                    )
                ]
            )

        #RECURSİVE
        #for path in files:
        #    if path.split("/")[-2] != ""

        if verbose:
            output = "- {}".format(path)
            for subPath in files:
                subPath = subPath.replace(path, "")
                output += "\n"
                for i in range(len(path.split("/"))):
                    output += "\t"
                output += "\t|- {}".format(subPath)
            print(output)
            print("{} files found".format(len(files)))

        return files

    except Exception as error:
        print(
            "An error occured while working in fileList function -> " + error.__str__(),
            # getframeinfo(currentframe()),
        )
    return None


def read_json(path):
    content = None
    with open(path) as json_file:
        content = json.load(json_file)
    return content


def show_image(
    source, title="Title", option="plot", cmap="gray", open_order=None, window=False
):

    if window:
        if isinstance(source, (list, tuple)):
            length_of_source_list = len(source)
            if open_order is None:
                for i in range(0, length_of_source_list):
                    show_image(
                        source[i], title=title, option=option, cmap=cmap, window=window
                    )
            elif open_order > 0:
                # https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly/46616645
                import matplotlib.pyplot as plot

                figure = plot.figure()

                columns = open_order
                if open_order >= length_of_source_list:
                    rows = 1
                else:
                    if (length_of_source_list % open_order) > 0:
                        rows = int(length_of_source_list / open_order) + 1
                    else:
                        rows = int(length_of_source_list / open_order)

                for i in range(1, length_of_source_list + 1):
                    figure.add_subplot(rows, columns, i)
                    plot.imshow(source[i - 1], cmap=cmap)
                plot.show()
        else:
            if option == "cv2":
                try:
                    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(title, 600, 600)

                    cv2.imshow(title, source)
                    cv2.waitKey()
                    cv2.destroyAllWindows()

                except Exception as error:
                    logging.error(
                        "An error occured while working in show_image function -> "
                        + error.__str__()
                    )
            elif option == "plot":
                try:
                    # w = 10
                    # h = 10

                    #fig = plt.figure(figsize=(8, 8))

                    # columns = 1
                    # rows = 1

                    # for i in range(1, columns * rows + 1):
                    #     fig.add_subplot(rows, columns, i)
                    #     plt.imshow(image, cmap=cmap)
                    # fig = plt.figure()
                    # fig.add_subplot(1, 1, 1)

                    plt.imshow(source, cmap=cmap)
                    plt.show()

                except Exception as error:
                    logging.error(
                        "An error occured while working in show_image function -> "
                        + error.__str__()
                    )
            else:
                logging.warning("Invalid option")

        # For skimage
        # plt.imshow(image, cmap='gray', interpolation='nearest')
        # plt.show()
    return 0


def load_from_json(path):
    data = None
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def dict_to_list(dict, is_only_value):
    dict_list = list()
    for key, value in dict.items():
        if is_only_value:
            dict_list.append(value)
        else:
            dict_list.append([key, value])
    return dict_list
"""


def draw_configs(path_directory, sources):
    draw_points = None

    files = list_files(
        path_directory, name="*", extensions=["json"], recursive=True, verbose=False
    )

    for path_file in files:
        draw_points = load_from_json(path_file)

        for image in sources:
            for line in draw_points:
                # Start coordinate, here (0, 0)
                # represents the top left corner of image
                start_point = (
                    int(draw_points["config"][0]["coord"][0]),
                    int(draw_points["config"][0]["coord"][1]),
                )

                # End coordinate, here (250, 250)
                # represents the bottom right corner of image
                end_point = (
                    int(draw_points["config"][0]["coord"][2]),
                    int(draw_points["config"][0]["coord"][3]),
                )

                # RED color in RGB
                color = (255, 0, 0)

                # Line thickness of 3 px
                thickness = 3

                image = cv2.line(image, start_point, end_point, color, thickness)


def alignImages(im1, im2, original_coordinate, sample_coordinate):

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    """
    # Extract location of good matches
    points1s = np.zeros((len(matches), 2), dtype=np.float32)
    points2s = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1s[i, :] = keypoints1[match.queryIdx].pt
        points1s[i, :] = keypoints2[match.trainIdx].pt
    """
    points1 = np.array(original_coordinate, dtype=np.float32)
    points2 = np.array(sample_coordinate, dtype=np.float32)

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


if __name__ == "__main__":

    # Read reference image
    refFilename = sys.argv[1]
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = sys.argv[2]
    print("Reading image to align : ", imFilename)
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    json_data = load_from_json(sys.argv[3])
    coordinates = dict_to_list(json_data, True)
    print("Coordinates: ", coordinates)

    original_coordinate = list()
    sample_coordinate = list()

    i = 1
    for coordinate in coordinates:
        if i % 2 == 0:
            sample_coordinate.append(coordinate)
        else:
            original_coordinate.append(coordinate)
        i += 1

    print("Aligning images ...")
    # Registered image will be resorted in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(im, imReference, original_coordinate, sample_coordinate)

    # Write aligned image to disk.
    outFilename = "aligned.jpg"
    print("Saving aligned image : ", outFilename)
    cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    print("Estimated homography : \n", h)

    source = [
        cv2.cvtColor(imReference, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(im, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(imReg, cv2.COLOR_BGR2RGB),
    ]

    path = sys.argv[3]
    draw_configs(path, source)

    show_image(
        source, title="Title", option="plot", cmap="gray", open_order=3, window=True
    )
