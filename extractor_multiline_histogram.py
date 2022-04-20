import numpy as np

def extract_Signal(img, startx, starty, endx, endy):
        start_point = (startx, starty)
        end_point = (endx, endy)
        
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

        currentColor = []
        for j in range(0, dx+dy, 1):
            currentColor.append((j, img[y0][x0]))

            e1 = e + dy
            e2 = e - dx
            if abs(e1) < abs(e2):
                x0 += ix
                e = e1
            else:
                y0 += iy
                e = e2
                
        return np.array(currentColor)