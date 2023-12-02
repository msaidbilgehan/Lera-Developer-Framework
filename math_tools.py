import math
import numpy as np
import secrets


def random_Bulk_Data(seed_number, start_range, end_range, number_of_data):
    secrets.SystemRandom().seed(seed_number)
    return [
        secrets.SystemRandom().randint(start_range, end_range)
        for i in range(0, number_of_data)
    ]


def random_Bulk_Data_Faster(seed_number, start_range, end_range, number_of_data):
    secrets.SystemRandom().seed(seed_number)
    return secrets.SystemRandom().sample(range(start_range, end_range), number_of_data)

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
    
    return new_x, new_y
