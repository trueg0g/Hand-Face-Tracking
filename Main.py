import cv2
import mediapipe as mp
import numpy as np
import random

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

gravity = 0.5
ground_level = 480
cube_count = 3

cubes = []
for _ in range(cube_count):
    cube = {
        'position': [random.randint(100, 300), random.randint(100, 200)],
        'velocity': [0, 0],
        'size': random.randint(30, 60),
        'color': (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
        'weight': random.uniform(0.4, 1.0),
        'rotation': 0,
        'rotation_speed': 0,
        'picked': False,
        'hand_index': None
    }
    cubes.append(cube)

holding_cubes = []
hold_distance_threshold = 50
release_threshold = 100

def is_hand_over_cube(hand_x, hand_y, cube_x, cube_y, cube_size):
    return cube_x < hand_x < cube_x + cube_size and cube_y < hand_y < cube_y + cube_size

def metallic_hand_effect(frame, landmarks, frame_width, frame_height):
    metallic_color = (192, 192, 192)
    highlight_color = (255, 255, 255)

    for lm in landmarks:
        x = int(lm.x * frame_width)
        y = int(lm.y * frame_height)

        cv2.circle(frame, (x, y), 10, metallic_color, -1)
        cv2.circle(frame, (x, y), 6, highlight_color, -1)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = hands.process(frame_rgb)
        face_results = face_mesh.process(frame_rgb)

        black_bg = np.zeros_like(frame)

        if hand_results.multi_hand_landmarks:
            for hand_index, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                finger_tip = hand_landmarks.landmark[8]
                finger_x = int(finger_tip.x * frame_width)
                finger_y = int(finger_tip.y * frame_height)

                for i, cube in enumerate(cubes):
                    distance = np.sqrt((cube['position'][0] - finger_x) ** 2 + (cube['position'][1] - finger_y) ** 2)

                    if not cube['picked'] and distance < hold_distance_threshold:
                        holding_cubes.append(i)
                        cube['picked'] = True
                        cube['hand_index'] = hand_index
                        cube['velocity'] = [0, 0]
                        cube['rotation_speed'] = 0

                for cube_index in holding_cubes:
                    cube = cubes[cube_index]
                    if cube['hand_index'] == hand_index:
                        distance = np.sqrt((cube['position'][0] - finger_x) ** 2 + (cube['position'][1] - finger_y) ** 2)
                        if distance > release_threshold:
                            cube['picked'] = False
                            holding_cubes.remove(cube_index)
                        else:
                            cube['position'] = [finger_x - cube['size'] // 2, finger_y - cube['size'] // 2]
                            cube['velocity'] = [(finger_x - cube['position'][0]) / 2, (finger_y - cube['position'][1]) / 2]
                            cube['rotation_speed'] = (cube['velocity'][0] ** 2 + cube['velocity'][1] ** 2) ** 0.5 / 20

                metallic_hand_effect(black_bg, hand_landmarks.landmark, frame_width, frame_height)

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=black_bg,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=black_bg,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=black_bg,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())

        for cube in cubes:
            if not cube['picked']:
                cube['velocity'][1] += gravity * cube['weight']
                cube['position'][1] += int(cube['velocity'][1])
                cube['position'][0] += int(cube['velocity'][0])
                cube['rotation'] += cube['rotation_speed']

                if cube['position'][1] + cube['size'] > ground_level:
                    cube['position'][1] = ground_level - cube['size']
                    cube['velocity'][1] *= -0.7

                if cube['position'][0] + cube['size'] > frame_width:
                    cube['position'][0] = frame_width - cube['size']
                    cube['velocity'][0] *= -0.7
                    cube['rotation_speed'] *= -1

                if cube['position'][0] < 0:
                    cube['position'][0] = 0
                    cube['velocity'][0] *= -0.7
                    cube['rotation_speed'] *= -1

            center = (cube['position'][0] + cube['size'] // 2, cube['position'][1] + cube['size'] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, cube['rotation'], 1)
            box = np.array([
                [-cube['size'] // 2, -cube['size'] // 2],
                [cube['size'] // 2, -cube['size'] // 2],
                [cube['size'] // 2, cube['size'] // 2],
                [-cube['size'] // 2, cube['size'] // 2]
            ])
            box = np.dot(rotation_matrix[:, :2], box.T).T + center
            box = box.astype(np.int32)

            cv2.fillConvexPoly(black_bg, box, cube['color'])

        cv2.imshow('GoG', black_bg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
