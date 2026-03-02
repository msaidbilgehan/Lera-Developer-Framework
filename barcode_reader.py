import cv2
import numpy as np
from pyzbar import pyzbar

from image_manipulation import draw_Rectangle


def decode_Barcodes(frame, show_detection=False, show_detection_thickness=2):

    if (frame.shape == 3):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    barcodes = pyzbar.decode(frame)
    list_barcodes = list()

    if barcodes:
        for barcode in barcodes:
            # x, y, w, h = barcode.rect
            barcode_text = barcode.data.decode('utf-8')

            if show_detection:
                x, y, w, h = barcode.rect
                draw_Rectangle(
                    frame,
                    (x, y),
                    (w, h),
                    color=(0, 255, 0),
                    thickness=show_detection_thickness
                )
                list_barcodes.append((barcode.type, barcode_text))

    else:
        list_barcodes.append((-1, -1))

    list_barcodes = np.array(list_barcodes)

    # show_image(frame, ['show'])
    # stdo(1, "Detected Barkod: {} | {}".format(barcode[:,0], barcode[:,1]))

    return list_barcodes, frame
