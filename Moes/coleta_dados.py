import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import csv
import time

DATA_PATH = "dados_gestos.csv"  
NUM_LANDMARKS = 21  # 

# --- Inicialização do MediaPipe ---
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
hands = vision.HandLandmarker.create_from_options(options)



def extrair_features(hand_landmarks):
    """
    Converte os landmarks da mão em uma lista simples de coordenadas normalizadas.
    """
    features = []

    for i in range(NUM_LANDMARKS):
        landmark_x = hand_landmarks[i].x
        landmark_y = hand_landmarks[i].y
        landmark_z = hand_landmarks[i].z
        features.append(landmark_x)
        features.append(landmark_y)
        features.append(landmark_z)

    return features

def salvar_dados(dados, nome_arquivo):
    """
    Salva a lista de dados em um arquivo CSV.
    """
    expected_header = []
    for i in range(NUM_LANDMARKS):
        expected_header += [f'x{i}', f'y{i}', f'z{i}']
    expected_header.append('label')

    file_exists = os.path.isfile(nome_arquivo)
    if file_exists:
        with open(nome_arquivo, 'r', newline='') as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
        if existing_header != expected_header:
            backup_name = nome_arquivo.replace('.csv', '_old.csv')
            os.replace(nome_arquivo, backup_name)
            print(f"Arquivo antigo renomeado para '{backup_name}' porque o cabeçalho estava incompatível.")
            file_exists = False

    with open(nome_arquivo, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(expected_header)
        writer.writerows(dados)

    print(f"\n{len(dados)} amostras salvas em '{nome_arquivo}'!")

# --- Coleta de Dados ---
if __name__ == "__main__":
    nome_gesto_left = input("Digite o nome do gesto para a MÃO ESQUERDA (ex: A_esq): ")
    nome_gesto_right = input("Digite o nome do gesto para a MÃO DIREITA (ex: B_dir): ")
    num_amostras = int(input("Digite o número de amostras a coletar (ex: 100): "))

    dados_coletados = []
    amostras_capturadas = 0
    cap = cv2.VideoCapture(0)

    print("\nPosicione as DUAS MÃOS na câmera. Pressione 's' para iniciar a coleta.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Inverte a imagem para um efeito de espelho, facilitando o posicionamento
        # frame = cv2.flip(frame, 1)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = hands.detect(mp_image)

        if results and results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Desenha as conexões entre os landmarks
                # Conexões da mão: [índice_início, índice_fim]
                conexoes = [
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
                
                # Desenha as linhas primeiro (para ficarem atrás dos pontos)
                for inicio, fim in conexoes:
                    x1 = int(hand_landmarks[inicio].x * frame.shape[1])
                    y1 = int(hand_landmarks[inicio].y * frame.shape[0])
                    x2 = int(hand_landmarks[fim].x * frame.shape[1])
                    y2 = int(hand_landmarks[fim].y * frame.shape[0])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Desenha os pontos dos landmarks
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        cv2.imshow("Coleta de Dados - Pressione 's' para iniciar", frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            # Contagem regressiva
            for i in range(3, 0, -1):
                print(f"Iniciando em {i}...")
                time.sleep(1)
            
            print("Coletando amostras...")
            while amostras_capturadas < num_amostras:
                ret, frame_coleta = cap.read()
                if not ret: break
                
                frame_coleta = cv2.flip(frame_coleta, 1)
                frame_rgb_coleta = cv2.cvtColor(frame_coleta, cv2.COLOR_BGR2RGB)
                
                # Converter para MediaPipe Image
                mp_image_coleta = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb_coleta)
                results_coleta = hands.detect(mp_image_coleta)

                # Garante que estamos coletando dados apenas quando AMBAS as mãos são detectadas
                if results_coleta and results_coleta.hand_landmarks and len(results_coleta.hand_landmarks) == 2:
                    for i, hand_landmarks in enumerate(results_coleta.hand_landmarks):
                        # Identifica se a mão é esquerda ou direita
                        handedness = results_coleta.handedness[i][0].category_name
                        
                        features = extrair_features(hand_landmarks)
                        if handedness == 'Left':
                            dados_coletados.append(features + [nome_gesto_left])
                        elif handedness == 'Right':
                            dados_coletados.append(features + [nome_gesto_right])
                    amostras_capturadas += 1
                    print(f"Amostra {amostras_capturadas}/{num_amostras} coletada.", end='\r')

            salvar_dados(dados_coletados, DATA_PATH)
            break 
        elif key == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()