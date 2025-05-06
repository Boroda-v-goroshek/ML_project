# IndustryGuard™
<img src="industryguard.gif" width=800>


## Введение

В рамках этого проекта происходит реализация обнаружения людей и элементов СИЗ на изображениях, с использованием модели YOLO11. Проект включает в себя использование инференса модели и телеграм-бота (в реализации)

## Структура репозитория

```plaintext
├── dataset/
│   ├── images
│   └── labels
│   └── data.yaml
├── app/
│   ├── main.py
│   └── video_inferens.py
├── weights/
│   ├── person_det_model.pt
│   └── yolo11n.pt
│   └── ppe_det_model.pt (пока в реализации)
└── inference.py
└── app.py
└── pascalVOC_to_yolo.py
├── README.MD
├── ppe_detect.ipynb
```

- **app:** Инференс модели
- **Weights** Сохраненные веса для модели
- **ppe_detect.ipynb:** Ноутбук с обучением модели

## Шаги

### 1. YOLOv11 Model Training

- **Модель детекции людей:**
  - Обучена на датасете из самых разных людей.
  - Epochs: 80, Batch size: 32.
   
     <img src="https://github.com/sabeer-k-s/YOLOv8-Person-PPE-Detection/blob/main/readme_images/5.png" width=375>
      <img src="https://github.com/sabeer-k-s/YOLOv8-Person-PPE-Detection/blob/main/readme_images/6.png" width=550>

- **Модель детекции СИЗ:**
  - Обучена на датасете из обрезанных изображений с людьми, с использованием первой модели.
  - Epochs: 80, Batch size: 32.
     
     <img src="https://github.com/sabeer-k-s/YOLOv8-Person-PPE-Detection/blob/main/readme_images/8.png" width=375>
      <img src="https://github.com/sabeer-k-s/YOLOv8-Person-PPE-Detection/blob/main/readme_images/9.png" width=550>

### 2. Этап инференса

- Обнаружение людей на изображении.
- Вырезание bbox человека и подача его в модель обнаружения СИЗ.
- Визуализация результатов с помощью OpenCV.

## Результаты
-
     <img src="https://github.com/sabeer-k-s/YOLOv8-Person-PPE-Detection/blob/main/readme_images/10.png" width=375>
      <img src="https://github.com/sabeer-k-s/YOLOv8-Person-PPE-Detection/blob/main/readme_images/11.png" width=550>
  <img src="https://github.com/sabeer-k-s/YOLOv8-Person-PPE-Detection/blob/main/readme_images/12.png" width=550>
  
---
