import cv2

from image_manipulation import threshold
from image_tools import save_image


def preprocess_of_Detect_Stain(
    image,
    # is_middle_object=False,
    # middle_object_roi=[],
    # threshold_config=[-1, -1],
    # is_line=False,
    # line_frame_preprocessed=None,
    is_stain_threshold=False,
    stain_threshold_config=[-1, -1],
    # stain_frame_preprocessed=None,
    # crop_ratio=5,
    sector_index='',
    # background_color='white',
    activate_debug_images=False
):
    """
    if (len(image.shape) == 3):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    """
    # image=stain_frame_preprocessed
    #

    """ 20.03.2025
    if is_middle_object:
        mask_image = image.copy()
        if background_color == 'white' or background_color == 'gray':
            mask_image[middle_object_roi[0]:middle_object_roi[1], middle_object_roi[2]:middle_object_roi[3]] = 255, 255, 255
        elif background_color == 'black' or background_color == 'piano-black':
            mask_image[middle_object_roi[0]:middle_object_roi[1], middle_object_roi[2]:middle_object_roi[3]] = 0, 0, 0
    else:
        mask_image = image

    #
    if background_color == 'white' or background_color == 'gray':
        color = (255,255,255)
    elif background_color == 'black' or background_color == 'piano-black':
        color = (0,0,0)
    temp_mask_image = mask_image.copy()
    fill_padded_image = draw_Rectangle(
        temp_mask_image,
        start_point=(0,0),
        end_point=(mask_image.shape[1], mask_image.shape[0]),
        color=color,
        thickness=20
    )

    #
    if is_line:
        # gray_image_fill_padded_image = cv2.cvtColor(fill_padded_image, cv2.COLOR_RGB2GRAY)
        # image_threshold = threshold(gray_image_fill_padded_image.copy(), configs=threshold_config)
        if background_color == 'white' or background_color == 'gray':
            padded_line_frame_preprocessed = cv2.copyMakeBorder(line_frame_preprocessed, 10,10,10,10, cv2.BORDER_CONSTANT, None, value=[0])
        elif background_color == 'black' or background_color == 'piano-black':
            padded_line_frame_preprocessed = cv2.copyMakeBorder(line_frame_preprocessed, 10,10,10,10, cv2.BORDER_CONSTANT, None, value=[0])

        _, _, draw_masking_image  = detect_Line_Object(
            line_eliminate_image=padded_line_frame_preprocessed.copy(),
            draw_masking_image=fill_padded_image.copy(),
            background_color=background_color,
            pano_title='ds',
            flag_activate_debug_images=activate_debug_images
        )
    else:
        draw_masking_image = fill_padded_image.copy()
    """

    #
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image
    if is_stain_threshold:
        image_threshold = threshold(gray_image.copy(), configs=stain_threshold_config)
    else:
        image_threshold = gray_image
    image_threshold = image_threshold[10: -10, 10: -10]  # 15.07.2025 Crop the image to remove borders

    if activate_debug_images:
        list_temp_save_image = [image, gray_image, image_threshold]
        filename = [str(sector_index)+"_0_image", str(sector_index)+"_1_gray_image", str(sector_index)+"_2_image_threshold"]
        save_image(list_temp_save_image, path="temp_files/detect_stains/preprocess_of_Detect_Stain", filename=filename, format="jpg")

    return image_threshold
