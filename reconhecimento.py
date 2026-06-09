import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle
import numpy as np

# --- Configurações e Carregamento do Modelo ---
MODEL_PATH = "C:\\Users\\Gustavo\\Desktop\\Synalizze - Sem Site\\Maos\\modelo_gestos.pkl"
LABEL_ENCODER_PATH = "C:\\Users\\Gustavo\\Desktop\\Synalizze - Sem Site\\Maos\\label_encoder.pkl"
NUM_LANDMARKS = 21

print("Carregando modelo...")
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
except FileNotFoundError:
    print("Erro: Arquivos de modelo não encontrados. Execute 'treinamento.py' primeiro.")
    exit()
    #Teste de PR para a IA gerar um sumário do que foi alterado no código, focando em inconsistências de sintaxe, uso de keywords e estrutura.    

print("Modelo carregado com sucesso!")

# --- Inicialização do MediaPipe (nova API) ---
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
hands = vision.HandLandmarker.create_from_options(options)

# Conexões para desenhar as linhas da mão
HAND_CONNECTIONS = [
    # Polegar: 0-1-2-3-4
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Indicador: 0-5-6-7-8
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Médio: 0-9-10-11-12
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Anelinho: 0-13-14-15-16
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Mindinho: 0-17-18-19-20
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Conexões horizontais da palma
    (5, 9), (9, 13), (13, 17)
]

def draw_rounded_rectangle(img, pt1, pt2, color, radius):
    """
    Desenha um retângulo com cantos arredondados.
    pt1: Canto superior esquerdo
    pt2: Canto inferior direito
    """
    x1, y1 = pt1
    x2, y2 = pt2

    # Desenha os 4 cantos (círculos preenchidos)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)

    # Desenha os retângulos de preenchimento
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)

def extrair_features(hand_landmarks):
    """
    Extrai as features da mão para a predição, da mesma forma que no treinamento.
    """
    features = []
    pulso_x = hand_landmarks[0].x
    pulso_y = hand_landmarks[0].y

    for i in range(NUM_LANDMARKS):
        landmark_x = hand_landmarks[i].x
        landmark_y = hand_landmarks[i].y
        features.append(landmark_x - pulso_x)
        features.append(landmark_y - pulso_y)

    return features

def desenhar_mao(frame, hand_landmarks):
    """
    Desenha os pontos e conexões da mão no frame.
    """
    # Desenha as conexões
    for inicio, fim in HAND_CONNECTIONS:
        x1 = int(hand_landmarks[inicio].x * frame.shape[1])
        y1 = int(hand_landmarks[inicio].y * frame.shape[0])
        x2 = int(hand_landmarks[fim].x * frame.shape[1])
        y2 = int(hand_landmarks[fim].y * frame.shape[0])
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Desenha os pontos
    for landmark in hand_landmarks:
        x = int(landmark.x * frame.shape[1])
        y = int(landmark.y * frame.shape[0])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

# --- Loop Principal ---
cap = cv2.VideoCapture(0)

# --- Configuração da Janela ---
WINDOW_NAME = "Reconhecimento de Gestos"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    results = hands.detect(mp_image)

    if results and results.hand_landmarks:
        
        for i, hand_landmarks in enumerate(results.hand_landmarks):
            
            desenhar_mao(frame, hand_landmarks)

            handedness = results.handedness[i][0].category_name

            features = extrair_features(hand_landmarks)
            prediction_numeric = model.predict([features])[0]
            predicted_label = le.inverse_transform([prediction_numeric])[0]

            
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            font_scale = 1.5
            font_thickness = 2
            text_color = (0, 0, 0)       
            bg_color = (255, 255, 255)   
            corner_radius = 15           
            padding = 10

            
            if handedness == 'Left':
                text = f"Esquerda: {predicted_label}"
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                
                rect_start = (10, 30)
                rect_end = (rect_start[0] + text_w + padding, rect_start[1] + text_h + padding)
                
                text_org = (rect_start[0] + padding // 2, rect_start[1] + text_h + padding // 2)
            else: 
                text = f"Direita: {predicted_label}"
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                rect_start = (frame.shape[1] - text_w - 20 - padding, 30)
                rect_end = (frame.shape[1] - 10, rect_start[1] + text_h + padding)
                text_org = (rect_start[0] + padding // 2, rect_start[1] + text_h + padding // 2)

            
            draw_rounded_rectangle(frame, rect_start, rect_end, bg_color, corner_radius)
            cv2.putText(frame, text, text_org, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(10) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()