import math
import numpy as np
from random import seed, randint, sample
from scipy.signal import find_peaks
from shapely.geometry import LineString, Point
from sympy import Point, Line
import cv2

# from stdo import stdo


def random_Bulk_Data(seed_number, start_range, end_range, number_of_data):
    seed(seed_number)
    return [
        randint(start_range, end_range)
        for i in range(0, number_of_data)
    ]

def random_Bulk_Data_Faster(seed_number, start_range, end_range, number_of_data):
    seed(seed_number)
    return sample(range(start_range, end_range), number_of_data)


def coordinate_Scaling(x, y, old_w, old_h, new_w=0, new_h=0, crop_x=0, crop_y=0, degree=0, task="RESIZE", is_dual=True):
    new_x, new_y = None, None
    if task == "90DEGREE_ROTATION":
        if degree == -90:
            new_x = y
            new_y = new_h - x
        elif degree == 90:
            new_y = x
            new_x = new_w - y

    elif task == "ANGULAR_ROTATION":
        degree = -degree
        if is_dual == False:
            new_x = []
            new_y = []
            (cx, cy) = (round(new_w/2), round(new_h/2))
            for i in range(len(x)):
                # It must be get in radian type: degree
                teg_x = round( np.cos(degree) * (x[i] - cx) - np.sin(degree) * (y[i] - cy) + cx )
                teg_y = round( np.sin(degree) * (x[i] - cx) + np.cos(degree) * (y[i] - cy) + cy )

                new_x.append(teg_x)
                new_y.append(teg_y)
            return np.array(new_x), np.array(new_y)
        else:
            new_x = round( np.cos(degree) * (x[i] - cx) - np.sin(degree) * (y[i] - cy) + cx )
            new_y = round( np.sin(degree) * (x[i] - cx) + np.cos(degree) * (y[i] - cy) + cy )


        """
        if is_dual == False:
            radian = math.radians(-degree)
            (cx, cy) = (round(new_w/2), round(new_h/2))
            new_x = []
            new_y = []
            for i in range(len(x)):
                v = (x[i]-cx, y[i]-cy)
                teg_x = v[0]*math.cos(radian) - v[1]*math.sin(radian)
                teg_y = v[0]*math.sin(radian) + v[1]*math.cos(radian)
                new_x.append(round(cx + teg_x))
                new_y.append(round(cy + teg_y))
            return np.array(new_x), np.array(new_y)

        else:
            radian = math.radians(-degree)
            (cx, cy) = (round(new_w/2), round(new_h/2))
            v = (x-cx, y-cy)
            new_x = v[0]*math.cos(radian) - v[1]*math.sin(radian)
            new_y = v[0]*math.sin(radian) + v[1]*math.cos(radian)
            new_x = round(cx + new_x)
            new_y = round(cy + new_y)
        """

    elif task == "CROP":
        if is_dual == False:

            # print("x,y:", x, y)

            if (old_w > new_w) and (old_h > new_h):
                new_x = []
                new_y = []
                for i in range(len(x)):
                    new_x.append(x[i] - crop_x)
                    new_y.append(y[i] - crop_y)
            else:
                new_x = []
                new_y = []
                for i in range(len(x)):
                    new_x.append(x[i] + crop_x)
                    new_y.append(y[i] + crop_y)
            # print("new_x, new_y:", new_x, new_y)
            return np.array(new_x), np.array(new_y)

        else:
            # print("x,y:", x, y)
            if (old_w > new_w) and (old_h > new_h):
                new_x = x - crop_x
                new_y = y - crop_y
            else:
                new_x = x + crop_x
                new_y = y + crop_y
            # print("new_x, new_y:", new_x, new_y)

    elif task == "RESIZE":
        if is_dual == False:
            new_x = []
            new_y = []
            for i in range(len(x)):
                new_x.append(round((x[i] / old_w) * new_w))
                new_y.append(round((y[i] / old_h) * new_h))
            return np.array(new_x), np.array(new_y)

        else:
            new_x = round((x / old_w) * new_w)
            new_y = round((y / old_h) * new_h)

    elif task == "HORIZONTAL_FLIP":
        if is_dual == False:
            new_x = []
            new_y = []
            for i in range(len(x)):
                new_x.append(round(old_w - x))
                new_y.append(round(y))
            return np.array(new_x), np.array(new_y)

        else:
            new_x = round(old_w - x)
            new_y = round(y)

    elif task == "VERTICAL_FLIP":
        if is_dual == False:
            new_x = []
            new_y = []
            for i in range(len(x)):
                new_x.append(round(x))
                new_y.append(round(old_h - y))
            return np.array(new_x), np.array(new_y)

        else:
            new_x = round(x)
            new_y = round(old_h - y)

    elif task == "PROJECTION_TO_XY_AXIS":
        if is_dual == False:
            new_x = []
            new_y = []
            for i in range(len(x)):

                new_x.append(int((x[i] * old_w) // (math.sqrt(math.pow(old_w, 2) + math.pow(old_h, 2)))))
                new_y.append(int((y[i] * old_w) // (math.sqrt(math.pow(old_w, 2) + math.pow(old_h, 2))))) # NOT WORK

            return np.array(new_x), np.array(new_y)

        else:
            new_x = int((x * old_w) // (math.sqrt(math.pow(old_w, 2) + math.pow(old_h, 2))))
            new_y = int((y * old_w) // (math.sqrt(math.pow(old_w, 2) + math.pow(old_h, 2))))

    return new_x, new_y

def fast_fourier_transform(image_gray):
    return np.fft.fft2(image_gray)

def fast_fourier_transform_shift(image_fft):
    return np.fft.fftshift(image_fft)

def fast_fourier_transform_invert(image_fft):
    return abs(np.fft.ifft2(image_fft))

def complex_array_to_image(image_fft, k):
    return np.array(
        np.maximum(
            0,
            np.minimum(
                np.log(np.abs(image_fft)) * k,
                255
            )
        ),
        dtype=np.uint8
    )

def complex_array_info(array_complex):
    # https://dsp.stackexchange.com/questions/22807/how-can-i-get-maximum-complex-value-from-an-array-of-complex-values524289-in-p
    return {
        "max_real": array_complex.real.max(),  # maximum real part
        "max_imag": array_complex.imag.max(),  # maximum imaginary part
        "max_abs": np.abs(array_complex).max(),  # maximum absolute value

        "min_real": array_complex.real.min(),  # minimum real part
        "min_imag": array_complex.imag.min(),  # minimum imaginary part
        "min_abs": np.abs(array_complex).min(),  # minimum absolute value
    }

def extraction_Pixel_Values(image, start_point=(0,0), end_point=(0,0)):
        x0 = start_point[0]
        y0 = start_point[1]
        x1 = end_point[0]
        y1 = end_point[1]
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        if x0 < x1:
            ix = 1
        else:
            ix = -1

        if y0 < y1:
            iy = 1
        else:
            iy = -1
        e = 0

        currentColor = list()
        for j in range(0, dx+dy, 1):
            currentColor.append((j, image[y0][x0]))

            e1 = e + dy
            e2 = e - dx
            if abs(e1) < abs(e2):
                x0 += ix
                e = e1
            else:
                y0 += iy
                e = e2
        currentColor = np.array(currentColor)
        return currentColor

def find_Peak_Points(image, signal, is_scale_x_axis=False, is_scale_y_axis=False):
    peaks, _ = find_peaks(signal[:,1], height=255//2)
    peak_indices = signal[:,0][peaks]

    if is_scale_x_axis or is_scale_y_axis:

        peaks, _ = coordinate_Scaling(
            x=peaks,
            y=signal[:,0],
            old_w=image.shape[1],
            old_h=image.shape[0],
            task="PROJECTION_TO_XY_AXIS",
            is_dual=False
        )

    peak_distances = np.diff(peaks)
    if len(peak_distances) > 0:
        peak_max = np.max(peak_distances)
    else:
        peak_max = 0
    return peak_max, peak_distances, peak_indices

def calculation_Determination(a, b):
    return (a[0] * b[1]) - (a[1] * b[0])

def line_Intersection(line1, line2, method='2'):
    if method == '1':
        for i in range(2):
            for j in range(2):
                if line1[i][j] <= 0:
                    line1[i][j] = 0
                elif line2[i][j] <= 0:
                    line2[i][j] = 0

        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        div = calculation_Determination(xdiff, ydiff)
        if div == 0:
            return False, False

        d = (calculation_Determination(*line1), calculation_Determination(*line2))
        x = calculation_Determination(d, xdiff) / div
        y = calculation_Determination(d, ydiff) / div

        return round(x), round(y)

    elif method == '2':
        line_string_1 = LineString([line1[0], line1[1]])
        line_string_2 = LineString([line2[0], line2[1]])

        int_pt = line_string_1.intersection(line_string_2)
        if int_pt.is_empty:
            return False, False
        else:
            return round(int_pt.x), round(int_pt.y)

    elif method == '3':
        # x1 = line1[0][0], y1 = line1[0][1]
        # x2 = line1[1][0], y2 = line1[1][1]
        # x3 = line2[0][0], y3 = line2[0][1]
        # x4 = line2[1][0], y4 = line2[1][1]


        """
        px = (
                (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4)
            ) / (
                (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
            )


        py = (
                (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4)
            ) / (
                (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
            )
        """


        px = (
              (line1[0][0]*line1[1][1]-line1[0][1] * line1[1][0])*(line2[0][0]-line2[1][0])-(line1[0][0]- line1[1][0])*(line2[0][0]*line2[1][1]-line2[0][1]*line2[1][0])
            ) / (
                (line1[0][0]- line1[1][0])*(line2[0][1]-line2[1][1])-(line1[0][1] -line1[1][1])*(line2[0][0]-line2[1][0])
            )
        py = (
              (line1[0][0]*line1[1][1]-line1[0][1] * line1[1][0])*(line2[0][1]-line2[1][1])-(line1[0][1] -line1[1][1])*(line2[0][0]*line2[1][1]-line2[0][1]*line2[1][0])
            ) / (
                 (line1[0][0]- line1[1][0])*(line2[0][1]-line2[1][1])-(line1[0][1] -line1[1][1])*(line2[0][0]-line2[1][0])
            )

        # stdo(1, "line_Intersection-method:{} center:({}, {})".format(method, px, py))

        if (px == float("nan")) or (py == float("nan")):
            return round(px), round(py)
        else:
            return False, False

    elif method == '4':
        line_string_1 = Line( Point(line1[0]), Point(line1[1]) )
        line_string_2 = Line( Point(line2[0]), Point(line2[1]) )

        is_parallel = line_string_1.is_parallel(line_string_2)
        if is_parallel:
            return False, False
        else:
            is_intersection = line_string_1.intersection(line_string_2)
            cross_coords = np.array(is_intersection).astype(np.int32)[0]

            return cross_coords[0], cross_coords[1]

def get_Optimal_Font_Scale(text, width, height):
    for scale in reversed(range(0, 60, 1)):
        (label_width, label_height), baseline = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)

        new_width = label_width
        new_height = label_height
        if (new_width <= width) and (new_height <= height):
            return scale/15
    return 1

def filter_Close_Points(points, distance_threshold=50, angle_threshold=2):
    filtered_points = []
    to_remove = set()

    def similarity_score(p1, p2):
        (x1, y1, angle1), (x2, y2, angle2) = p1, p2
        distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        angle_diff = abs(angle1 - angle2)

        norm_distance = distance / distance_threshold  # Normalize distance
        norm_angle = angle_diff / angle_threshold  # Normalize angle difference

        return norm_distance + norm_angle  # Combined score

    count_i = 0
    for i in range(len(points)):
        if i in to_remove:
            continue

        count_j = 0
        for j in range(i + 1, len(points)):
            if j in to_remove:
                continue

            score = similarity_score(points[i], points[j])

            x1, y1, angle1 = points[i]
            x2, y2, angle2 = points[j]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if (score < 100) and (distance < distance_threshold):  # Adjust threshold if necessary
                to_remove.add(j)  # Remove second point if similar
                # stdo(1, "[{}][{}] score:{:.2f} | distance:{:.2f} - FAIL".format(count_i, count_j, score, distance))
            # else:
                # stdo(1, "[{}][{}] score:{:.2f} | distance:{:.2f} - STAY".format(count_i, count_j, score, distance))
            count_j += 1
        count_i += 1

    filtered_points = [point for i, point in enumerate(points) if i not in to_remove]

    return filtered_points

def filter_Close_Points_2(points, distance_threshold=50):
        filtered_points = []
        to_remove = set()

        count_i = 0
        for i in range(len(points)):
            if i in to_remove:
                continue

            count_j = 0
            for j in range(i + 1, len(points)):
                if j in to_remove:
                    continue

                x1, y1 = points[i]
                x2, y2 = points[j]
                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                if (distance < distance_threshold):
                    to_remove.add(j)
                    # stdo(1, "[{}][{}] distance:{:.2f} | ({:.1f},{:.1f})({:.1f},{:.1f}) - FAIL".format(count_i, count_j, distance, x1, y1, x2, y2))
                # else:
                    # stdo(1, "[{}][{}] distance:{:.2f} | ({:.1f},{:.1f})({:.1f},{:.1f}) - STAY".format(count_i, count_j, distance, x1, y1, x2, y2))

                count_j += 1
            count_i += 1

        filtered_points = [point for i, point in enumerate(points) if i not in to_remove]
        return filtered_points

def filter_Closest_Bottom_Points(positions):
    left_top = [p for p in positions if p[2] == "Left-Top"]
    left_bottom = [p for p in positions if p[2] == "Left-Bottom"]
    right_top = [p for p in positions if p[2] == "Right-Top"]
    right_bottom = [p for p in positions if p[2] == "Right-Bottom"]

    if not left_top or not right_top or not left_bottom or not right_bottom:
        return positions

    left_top_y = np.mean([float(p[1]) for p in left_top])  # Ortalama Y değeri al
    right_top_y = np.mean([float(p[1]) for p in right_top])

    min_diff = float("inf")
    best_pair = None

    for lb in left_bottom:
        for rb in right_bottom:
            diff = abs((float(lb[1]) - left_top_y) - (float(rb[1]) - right_top_y))
            if diff < min_diff:
                min_diff = diff
                best_pair = [lb, rb]

    if best_pair:
        return left_top + best_pair + right_top
    else:
        return positions

def get_Angles_Of_Horizontal_Lines(points):
    points = sorted(points, key=lambda p: p[0])  # Sort by x coordinate
    left_points = sorted(points[:2], key=lambda p: p[1])  # Top-bottom order
    right_points = sorted(points[2:], key=lambda p: p[1])  # Top-bottom order

    # stdo(1, "get_Angles_Of_Horizontal_Lines: left_points:{} | right_points:{}".format(left_points, right_points))

    if len(right_points) == 2 and len(left_points) == 2:
        angle_rad_top = np.arctan2(right_points[0][1]-left_points[0][1], right_points[0][0]-left_points[0][0]) # np.arctan2(y2 - y1, x2 - x1)
        angle_rad_bottom = np.arctan2(right_points[1][1]-left_points[1][1], right_points[1][0]-left_points[1][0]) # np.arctan2(y2 - y1, x2 - x1)
    else:
        angle_rad_top = 0
        angle_rad_bottom = 0

    return angle_rad_top, angle_rad_bottom

def determine_Line_Position(points, image_width, image_height):
    positions = []

    for x, y in points:
        if x < image_width / 3:
            position = "Left"
        elif x > 2 * image_width / 3:
            position = "Right"
        else:
            position = "Center"

        if y < image_height / 3:
            position += "-Top"
        elif y > 2 * image_height / 3:
            position += "-Bottom"
        else:
            position += "-Middle"

        positions.append((x, y, position))

    return positions

def compute_Contour_Thickness(id, cnt):
    """ Konturun minimum genişliğini hesaplar. """
    # Konturun en küçük alan kaplayan dikdörtgenini bul
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.intp(box)  # Köşe noktalarını tam sayı yap

    # Uzaklıkları hesapla (dört kenarın uzunlukları)
    edge_lengths = [
        np.linalg.norm(box[i] - box[(i + 1) % 4]) for i in range(4)
    ]

    # stdo(1, "[{}]: rect:{} | box:{} | edge_lengths:{}".format(id, rect, box, edge_lengths))

    # Minimum kenar uzunluğunu döndür
    return rect, min(edge_lengths)

def compute_Contour_Thickness_2(cnt, img_shape):
    """ Konturun gerçek minimum genişliğini hesaplar. """
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, 1)  # Konturu çiz (1 piksel kalınlığında)

    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)  # Uzaklık haritası
    min_thickness = np.min(dist_transform[mask == 255]) * 2  # En küçük genişlik

    # stdo(1, "[{}]: dist_transform:{}".format(dist_transform))

    return min_thickness

def get_Degree_Of_Spine_Angle(mid_shoulder, mid_hip):
    dx = mid_shoulder[0] - mid_hip[0]
    dy = mid_shoulder[1] - mid_hip[1]

    if dx == 0 and dy == 0:
        return 0.0

    angle_rad = math.atan2(dx, -dy)

    return abs(math.degrees(angle_rad))

def is_Bbox_Inside_Other_Bbox(big_bbox, small_bbox):
    big_x, big_y, big_w, big_h = big_bbox
    small_x, small_y, small_w, small_h = small_bbox

    # Büyük dikdörtgenin sınırları
    big_x2 = big_x + big_w
    big_y2 = big_y + big_h

    # Küçük dikdörtgenin köşe noktaları
    small_x2 = small_x + small_w
    small_y2 = small_y + small_h

    return (
        small_x >= big_x and
        small_y >= big_y and
        small_x2 <= big_x2 and
        small_y2 <= big_y2
    )

def point_In_Bbox_Match(point, bboxes):
    px, py = point
    for name, bbox in bboxes.items():

        if ("startx" in bbox) and ("starty" in bbox) and ("endx" in bbox) and ("endy" in bbox):

            if bbox["startx"] <= px <= bbox["endx"] and bbox["starty"] <= py <= bbox["endy"]:
                return name
        else:
            return None

    return None  # Hiç eşleşme yoksa

def bbox_IOU(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def int_To_Dot_String(value):
    """
    Converts an integer to a string with a dot as a decimal separator.

    Args:
        value (int): The integer to convert.

    Returns:
        str: The string representation of the integer with a dot.
    """
    s = str(value).zfill(3)  # En az 3 haneli yap (0,01 için)
    return f"{s[:-3]}.{s[-3:]}" if len(s) > 2 else f"0.{s[-3:]}"

def int_To_Comma_String(value):
    """
    Converts an integer to a string with a comma as a decimal separator.

    Args:
        value (int): The integer to convert.

    Returns:
        str: The string representation of the integer with a comma.
    """
    s = str(value).zfill(3)  # En az 3 haneli yap (0,01 için)
    return f"{s[:-3]},{s[-3:]}" if len(s) > 2 else f"0,{s[-3:]}"

def normalize_Comma_to_Dot_Float(value):
    # String'e çevir, virgülü noktaya çevir, sonra float'a dönüştür
    return float(str(value).replace(",", "."))
