
import time
import os
import cv2
import mediapipe as mp

use_solutions = hasattr(mp, 'solutions')
use_tasks = False
vision = None
python = None

try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    use_tasks = True
except ImportError:
    use_tasks = False

contador = 0
width, height = 250, 250
capturando_crops = False
crops_restantes = 0
hand1_pos_inicial = None
hand2_pos_inicial = None

WINDOW_NAME = "Mao - Reconhecimento"
cap = cv2.VideoCapture(0)

if use_solutions:
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
elif use_tasks:
    models_dir = os.path.dirname(__file__)
    hand_model_path = os.path.join(models_dir, 'hand_landmarker.task')

    if not os.path.isfile(hand_model_path):
        raise FileNotFoundError(
            f"Hand model not found: {hand_model_path}.\n"
            "Download or copy 'hand_landmarker.task' into the 'Moes' folder, or install a mediapipe package that supports mp.solutions."
        )

    base_options = python.BaseOptions(model_asset_path=hand_model_path)
    hand_options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hands = vision.HandLandmarker.create_from_options(hand_options)
else:
    raise ImportError(
        'Nenhuma API MediaPipe suportada encontrada.\n'
        'Instale um pacote mediapipe com mp.solutions ou adicione um modelo hand_landmarker.task ao diretório Moes.'
    )

is_fullscreen = True
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def draw_landmarks(frame, landmarks, connections, color=(0, 165, 255), radius=1, thickness=1):
    h, w = frame.shape[:2]
    for start, end in connections:
        start_pt = (int(landmarks[start].x * w), int(landmarks[start].y * h))
        end_pt = (int(landmarks[end].x * w), int(landmarks[end].y * h))
        cv2.line(frame, start_pt, end_pt, color, thickness)
    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), radius, color, -1)


def get_hand_landmarks_list(results):
    if use_solutions:
        return results.multi_hand_landmarks
    return getattr(results, 'hand_landmarks', None)


while True:
    ok, frame = cap.read()
    if not ok:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if use_solutions:
        results = hands.process(img_rgb)
        hand_landmarks_list = results.multi_hand_landmarks
        hand_connections = mp_hands.HAND_CONNECTIONS
    else:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = hands.detect(mp_image)
        hand_landmarks_list = getattr(results, 'hand_landmarks', None)
        hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

    if hand_landmarks_list:
        for hand_landmarks in hand_landmarks_list:
            draw_landmarks(
                frame,
                hand_landmarks,
                hand_connections,
                color=(0, 255, 0),
                radius=2,
                thickness=2,
            )

    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC para sair
        break
    elif key == ord('f'):  # Alterna tela cheia
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    elif key == ord('s'):  # tecla "s" para iniciar captura de 60 crops
        time.sleep(0.5)
        os.makedirs("diretorio_hand1", exist_ok=True)
        os.makedirs("diretorio_hand2", exist_ok=True)
        hand_landmarks_list = get_hand_landmarks_list(results)
        if hand_landmarks_list:
            h, w, _ = frame.shape
            hand_landmarks1 = hand_landmarks_list[0]
            ponto9_1 = hand_landmarks1[9]
            x_init1 = int(ponto9_1.x * w)
            y_init1 = int(ponto9_1.y * h)
            hand1_pos_inicial = (x_init1, y_init1)
            if len(hand_landmarks_list) > 1:
                hand_landmarks2 = hand_landmarks_list[1]
                ponto9_2 = hand_landmarks2[9]
                x_init2 = int(ponto9_2.x * w)
                y_init2 = int(ponto9_2.y * h)
                hand2_pos_inicial = (x_init2, y_init2)
            else:
                hand2_pos_inicial = None
            capturando_crops = True
            crops_restantes = 60

    if capturando_crops and crops_restantes > 0:
        hand_landmarks_list = get_hand_landmarks_list(results)
        if hand_landmarks_list and hand1_pos_inicial is not None:
            h, w, _ = frame.shape
            min_dist1 = None
            hand1_landmarks = None
            min_dist2 = None
            hand2_landmarks = None
            for hand_landmarks in hand_landmarks_list:
                ponto9 = hand_landmarks[9]
                x = int(ponto9.x * w)
                y = int(ponto9.y * h)
                dist1 = (x - hand1_pos_inicial[0])**2 + (y - hand1_pos_inicial[1])**2
                if min_dist1 is None or dist1 < min_dist1:
                    min_dist1 = dist1
                    hand1_landmarks = hand_landmarks
                if hand2_pos_inicial is not None:
                    dist2 = (x - hand2_pos_inicial[0])**2 + (y - hand2_pos_inicial[1])**2
                    if min_dist2 is None or dist2 < min_dist2:
                        min_dist2 = dist2
                        hand2_landmarks = hand_landmarks

            if hand1_landmarks is not None:
                ponto9 = hand1_landmarks[9]
                x = int(ponto9.x * w)
                y = int(ponto9.y * h)
                x1 = max(x - width // 2, 0)
                y1 = max(y - height // 2, 0)
                x2 = min(x + width // 2, w)
                y2 = min(y + height // 2, h)
                mao_crop = frame[y1:y2, x1:x2]
                nome_arquivo = f"diretorio_hand1/hand1_mao_{contador}.png"
                cv2.imwrite(nome_arquivo, mao_crop)
                print(f"Screenshot salva: {nome_arquivo}")

            if hand2_landmarks is not None and hand2_pos_inicial is not None:
                ponto9 = hand2_landmarks[9]
                x = int(ponto9.x * w)
                y = int(ponto9.y * h)
                x1 = max(x - width // 2, 0)
                y1 = max(y - height // 2, 0)
                x2 = min(x + width // 2, w)
                y2 = min(y + height // 2, h)
                mao_crop = frame[y1:y2, x1:x2]
                nome_arquivo = f"diretorio_hand2/hand2_mao_{contador}.png"
                cv2.imwrite(nome_arquivo, mao_crop)
                print(f"Screenshot salva: {nome_arquivo}")

            contador += 1
        crops_restantes -= 1
        if crops_restantes == 0:
            capturando_crops = False
            hand1_pos_inicial = None
            hand2_pos_inicial = None

cap.release()
cv2.destroyAllWindows()
