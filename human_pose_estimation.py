# import os
import time
# import math
from datetime import datetime

import cv2
import cvzone
# import numpy as np
# import torch

import libs
# from tools import load_from_json
# from image_manipulation import draw_Rectangle, draw_Text, get_Bbox_Position_Into_Image, color_Range_Mask_Using_Palette
# from image_tools import save_image
# from stdo import stdo
from math_tools import get_Degree_Of_Spine_Angle  # , coordinate_Scaling


id_map = {}
next_id = 0


def get_mapped_id(track_id):
    global next_id
    if track_id not in id_map:
        id_map[track_id] = next_id
        next_id += 1
    return id_map[track_id]


def human_Pose_Estimation(image, process_methods=None, device=None, body_node_connections=[], face_node_indices=[], face_blurring_params=[],
                          cam_id=0, cam_ip='192.168.1.64', flag_activate_debug_images=False):

    # Per-camera workstation regions
    workstations_by_cam = {
        "192.168.60.71": {  # IP Camera 1
            1: (473, 426, 679, 673),
            2: (838, 531, 1086, 797)
        },
        "192.168.60.72": {  # IP Camera 2
            3: (547, 220, 751, 460),
            4: (1004, 173, 1222, 382)
        },
        "192.168.60.70": {  # IP Camera 3
            5: (563, 371, 825, 762),
            6: (869, 371, 1135, 762)
        },
    }
    workstations = workstations_by_cam.get(cam_ip, {})

    start_time = time.time()

    # --- POSE ESTIMATION & ANALYSIS --- #
    start_pred = time.time()
    results = process_methods.track(
        image,  # Orijinal frame üzerinde tracking
        persist=True,
        imgsz=640,
        conf=0.7,
        iou=0.3,
        device=device,
        max_det=10,
        verbose=False,
    )
    stop_pred = time.time() - start_pred

    start_draw = time.time()

    person_data = []

    display_image = image.copy()

    for ws_id, (wx1, wy1, wx2, wy2) in workstations.items():
        cv2.rectangle(display_image, (wx1, wy1), (wx2, wy2), (255, 255, 0), 2)
        cvzone.putTextRect(display_image, f'WS {ws_id}',
                           [wx1 + 5, wy1 - 10], thickness=2, scale=1, colorR=(255,255,0))

    # working_image = image.copy()
    face_history = {}
    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else []

        for idx, (person_kpts, bbox) in enumerate(zip(keypoints.data, boxes)):
            points = person_kpts
            # ensure shoulder & hip confidence
            required_kps = [5, 6, 11, 12]
            if not all(points[kp][2] > 0.5 for kp in required_kps):
                continue

            mid_shoulder = ((points[5][0] + points[6][0]) / 2,
                            (points[5][1] + points[6][1]) / 2)
            mid_hip = (
                (points[11][0] + points[12][0]) / 2,
                (points[11][1] + points[12][1]) / 2
            )

            """spine_angle  = get_Degree_Of_Spine_Angle(mid_shoulder, mid_hip)

            posture_status = "Good Posture"
            color = (0, 255, 0)
            if spine_angle > 27:
                posture_status = "Bad Posture!"
                color = (0, 0, 255)"""

            """dx = mid_shoulder[0] - mid_hip[0]
            dy = mid_shoulder[1] - mid_hip[1]
            spine_slope = abs(dx / (dy + 1e-5))

            posture_status = "Good Posture"
            color = (0, 255, 0)
            if spine_slope > 0.5:
                posture_status = "Bad Posture!"
                color = (0, 0, 255)"""


            # spine eğimi
            spine_angle = get_Degree_Of_Spine_Angle(mid_shoulder, mid_hip)
            spine_bent = spine_angle > 35

            # burun-kalça hizası (baş öne mi?)
            head_forward = False
            if points[0][2] > 0.5 and points[11][2] > 0.5 and points[12][2] > 0.5:
                nose_x = points[0][0]
                hip_x = (points[11][0] + points[12][0]) / 2
                # if abs(nose_x - hip_x) > 80:
                #     head_forward = True
                frame_width = image.shape[1]
                if abs(nose_x - hip_x) > frame_width * 0.15:
                    head_forward = True

            # kulak-omuz mesafesi (yana eğilme var mı?)
            side_lean = False
            ear_shoulder_dist = None  # <-- güvenli başlatma
            if points[3][2] > 0.5 and points[5][2] > 0.5:
                head_height = abs(points[0][1] - points[11][1])
                ear_shoulder_dist = abs(points[3][1] - points[5][1])
                if ear_shoulder_dist < 0.1 * head_height:
                    side_lean = True

            # kenarda mı? (kamera merkezine çok uzaksa es geç)
            image_center_x = image.shape[1] // 2
            is_edge = abs(mid_shoulder[0] - image_center_x) > image.shape[1] * 0.48

            # Postür Kararı
            posture_status = "Good Posture"
            color = (0, 255, 0)

            if not is_edge and (spine_bent or head_forward or side_lean):
                posture_status = "Bad Posture!"
                color = (0, 0, 255)

            # stdo(1, f"Person {idx} - Spine Angle: {spine_angle:.2f}, Posture: {posture_status}, Color: {color}, Head Forward: {head_forward}, Side Lean: {side_lean}, Is Edge: {is_edge}, Spine Bent: {spine_bent}")


            # draw bounding box & ID
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = map(int, bbox)
            track_id = track_ids[idx] if idx < len(track_ids) else idx
            mapped_id = get_mapped_id(track_id)

            # bbox merkezi
            center_x = (bbox_x1 + bbox_x2) // 2
            center_y = (bbox_y1 + bbox_y2) // 2

            # workstation kontrolü
            assigned_id = None
            for ws_id, (wx1, wy1, wx2, wy2) in workstations.items():
                if wx1 <= center_x <= wx2 and wy1 <= center_y <= wy2:
                    assigned_id = ws_id
                    break

            # sadece iş istasyonu içine düşenleri göster
            if assigned_id is None:
                continue  # bu kişiyi hiç çizme

            display_id = assigned_id

            # ekrana ID yaz
            cvzone.cornerRect(display_image, [bbox_x1, bbox_y1, bbox_x2 - bbox_x1, bbox_y2 - bbox_y1], l=30, rt=6, colorC=(0,255,0))
            cvzone.putTextRect(display_image, f'Person {int(display_id)}', [bbox_x1 + 8, bbox_y1 - 12], thickness=2, scale=2)

            # draw skeleton body_node_connections
            for (start, end) in body_node_connections:
                if start >= len(points) or end >= len(points):
                    continue
                if points[start][2] > 0.5 and points[end][2] > 0.5:
                    start_pt = tuple(points[start][:2].astype(int))
                    end_pt = tuple(points[end][:2].astype(int))
                    cv2.line(display_image, start_pt, end_pt, (255,0,0), 2)

            # keypoints
            valid_points = points[points[:,2] > 0.5]
            for px, py, conf in points:
                if conf > 0.7:
                    cv2.circle(display_image, (int(px), int(py)), 5, (0,255,0), cv2.FILLED)

            # draw spine line
            cv2.line(display_image,
                     (int(mid_hip[0]), int(mid_hip[1])),
                     (int(mid_shoulder[0]), int(mid_shoulder[1])),
                     color, 4)

            # angle & status text
            cvzone.putTextRect(display_image, f"Angle: {spine_angle:.1f}",
                               (int(mid_hip[0] - 50), int(mid_hip[1] - 60)),
                               thickness=2, scale=1.5, colorR=color,
                               offset=10, border=cv2.BORDER_CONSTANT)
            cvzone.putTextRect(display_image, posture_status,
                               (int(mid_hip[0] - 50), int(mid_hip[1] - 30)),
                               thickness=2, scale=1.5, colorR=color,
                               offset=10, border=cv2.BORDER_CONSTANT)

            # önce keypoint’lerle tespit var mı bak
            valid_face_points = []
            for i in face_node_indices:
                if i < len(person_kpts) and person_kpts[i][2] > 0.3:
                    valid_face_points.append(person_kpts[i][:2])

            used_coords = None
            if valid_face_points:
                # gerçekte tespit edilen bölge
                x_coords = [int(p[0]) for p in valid_face_points]
                y_coords = [int(p[1]) for p in valid_face_points]

                # face_blurring_params[0] = face_blur_padding
                # face_blurring_params[1] = self_face_limit_width
                # face_blurring_params[2] = face_limit_height
                # face_blurring_params[3] = face_max_miss_frames
                x1 = max(0, min(x_coords) - face_blurring_params[0])
                y1 = max(0, min(y_coords) - face_blurring_params[0])
                x2 = min(image.shape[1], max(x_coords) + face_blurring_params[0])
                y2 = min(image.shape[0], max(y_coords) + face_blurring_params[0])

                # anlamsız büyük bölgeyi atla
                if (x2-x1) <= face_blurring_params[1] and (y2-y1) <= face_blurring_params[2]:
                    used_coords = (x1, y1, x2, y2)
                    # history güncelle
                    face_history[track_id] = {'coords': used_coords, 'miss': 0}
                else:
                # tespit yoksa, history’de varsa ve miss sayısı toleranstan küçükse
                    if track_id in face_history and face_history[track_id]['miss'] < face_blurring_params[3]:
                        used_coords = face_history[track_id]['coords']
                        face_history[track_id]['miss'] += 1
                    else:
                        # history’de çok bekledi veya hiç yoksa temizle
                        face_history.pop(track_id, None)

                # blur uygula
                # if used_coords:
                #     x1, y1, x2, y2 = used_coords
                #     face_roi = image[y1:y2, x1:x2]
                #     if face_roi.size > 0:
                #         kw = max(11, (int((x2-x1)*0.3)//2)*2 + 1)
                #         blurred_face = cv2.GaussianBlur(face_roi, (kw, kw), 0)
                #         display_image[y1:y2, x1:x2] = blurred_face

            """person_data.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'person_id': int(track_id),
                'person_bbox': (bbox_x1, bbox_y1, bbox_x2, bbox_y2),
                # 'keypoints': valid_points,
                #'spine_angle': spine_angle,
                'posture_status': posture_status,
                'cam_id': cam_id,
            })"""
            person_data.append([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                cam_ip,
                int(display_id),
                str([bbox_x1, bbox_y1, bbox_x2, bbox_y2]),
                posture_status
            ])

    stop_draw = time.time() - start_draw

    stop_time = (time.time() - start_time) * 1000
    # stdo(1, "HPE [{}]: T:{:.3f}ms - Pred:{:.3f} - Draw:{:.3f}".format(
    #     str(cam_id),
    #     stop_time,
    #     stop_pred,
    #     stop_draw
    # ))
    result_dashboard = "T:{:.3f}ms - Pred:{:.3f} - Draw:{:.3f}".format(
        stop_time,
        stop_pred,
        stop_draw
    )

    return display_image, result_dashboard, person_data