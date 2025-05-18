import os
import cv2
import tempfile

import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import streamlit as st

def load_model(model_path):
    return YOLO(model_path)

def check_ppe_completeness(ppe_labels):
    required_ppe = {'helmet', 'vest', 'gloves'}
    detected_ppe = set(ppe_labels)
    return required_ppe.issubset(detected_ppe)

def perform_inference_on_video(video_path, person_model_path, ppe_model_path):
    person_model = load_model(person_model_path)
    ppe_model = load_model(ppe_model_path)

    class_colors = {
        "person": (0, 255, 0),  # Green
        "helmet": (255, 0, 0),  # Blue
        "gloves": (0, 0, 255),  # Red
        "vest": (0, 255, 255),  # Yellow
        "glasses": (255, 255, 0)  # Cyan
    }

    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_output:
        output_path = temp_output.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Конфигурация трекинга
    MAX_TRACK_AGE = 30  # Максимальное количество фреймов "памяти" для потерянных треков
    DETECT_INTERVAL = 10  # Интервал проверки комплектности СИЗ
    current_frame_num = 0  # Счетчик фреймов
    
    # Словарь для хранения информации о треках
    tracks_info = defaultdict(lambda: {
        'ppe_set': set(),  # Множество найденных СИЗ
        'frames_count': 0,  # Счетчик фреймов
        'full_ppe': False,  # Статус комплектности
        'last_seen': 0,  # Номер фрейма, когда трек последний раз видели
        'first_seen': 0,  # Номер фрейма первого обнаружения
        'color': (0, 0, 0)  # Уникальный цвет для визуализации
    })

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_num += 1
        original_frame = frame.copy()

        # Детекция и трекинг людей
        person_results = person_model.track(
            source=frame,
            persist=True,
            tracker="botsort.yaml",
            conf=0.5,
            iou=0.5
        )
        
        # Очистка старых треков
        active_track_ids = []
        if person_results[0].boxes.id is not None:
            active_track_ids = person_results[0].boxes.id.cpu().numpy().astype(int).tolist()
        
        # Удаляем треки, которые не видели слишком долго
        to_delete = []
        for track_id in list(tracks_info.keys()):
            if (track_id not in active_track_ids and 
                current_frame_num - tracks_info[track_id]['last_seen'] > MAX_TRACK_AGE):
                to_delete.append(track_id)
        for track_id in to_delete:
            del tracks_info[track_id]

        if person_results[0].boxes.id is not None:
            person_bboxes = person_results[0].boxes.xyxy.cpu().numpy()
            person_track_ids = person_results[0].boxes.id.cpu().numpy().astype(int)

            for person_box, track_id in zip(person_bboxes, person_track_ids):
                # Инициализация нового трека
                if track_id not in tracks_info:
                    tracks_info[track_id] = {
                        'ppe_set': set(),
                        'frames_count': 0,
                        'full_ppe': False,
                        'last_seen': current_frame_num,
                        'first_seen': current_frame_num,
                        'color': tuple((int(x) for x in np.random.randint(0, 255, 3)))
                    }
                else:                    
                    tracks_info[track_id]['last_seen'] = current_frame_num

                x1, y1, x2, y2 = map(int, person_box[:4])
                cropped_frame = original_frame[y1:y2, x1:x2]

                # Детекция СИЗ
                ppe_results = ppe_model.predict(cropped_frame)
                ppe_labels = [ppe_results[0].names[int(cls)] for cls in ppe_results[0].boxes.cls.cpu().numpy()]
                
                # Обновляем множество СИЗ
                tracks_info[track_id]['ppe_set'].update(ppe_labels)
                tracks_info[track_id]['frames_count'] += 1

                # Проверка комплектности по интервалу
                if tracks_info[track_id]['frames_count'] >= DETECT_INTERVAL:
                    tracks_info[track_id]['full_ppe'] = check_ppe_completeness(tracks_info[track_id]['ppe_set'])
                    tracks_info[track_id]['frames_count'] = 0
                    tracks_info[track_id]['ppe_set'] = set()  # Сброс для нового интервала

                # Корректировка координат СИЗ
                ppe_bboxes = ppe_results[0].boxes.xyxy.cpu().numpy()
                adjusted_ppe_boxes = []
                for ppe_box in ppe_bboxes:
                    adjusted_ppe_boxes.append([
                        ppe_box[0] + x1,
                        ppe_box[1] + y1,
                        ppe_box[2] + x1,
                        ppe_box[3] + y1
                    ])
                    
                track_color = tracks_info[track_id]['color']
                cv2.rectangle(frame, (x1, y1), (x2, y2), track_color, 2)
                
                # Информация о треке
                track_info = f"ID: {track_id} | Age: {current_frame_num - tracks_info[track_id]['first_seen']}f"
                cv2.putText(frame, track_info, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, track_color, 1)
                
                # Статус комплектности
                status = "FULL PPE" if tracks_info[track_id]['full_ppe'] else "MISSING PPE"
                status_color = (0, 255, 0) if tracks_info[track_id]['full_ppe'] else (0, 0, 255)
                cv2.putText(frame, status, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

                # Отрисовка СИЗ
                for ppe_box, ppe_label in zip(adjusted_ppe_boxes, ppe_labels):
                    px1, py1, px2, py2 = map(int, ppe_box)
                    cv2.rectangle(frame, (px1, py1), (px2, py2), class_colors.get(ppe_label, (255, 255, 255)), 1)
                    cv2.putText(frame, ppe_label, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, class_colors.get(ppe_label, (255, 255, 255)), 1)
        
        out.write(frame)

    cap.release()
    out.release()

    return output_path


def main():
    st.title("РовныеБРО™")
    st.subheader("PPE detector")

    person_model_path = "person_det_model.pt"
    ppe_model_path = "ppe_best.pt"

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(uploaded_file.read())
            temp_video_path = temp_video.name

        if st.button("Process Video"):
            with st.spinner("Processing video..."):
                output_video = perform_inference_on_video(temp_video_path, person_model_path, ppe_model_path)
                if output_video:
                    st.video(output_video)

if __name__ == "__main__":
    main()