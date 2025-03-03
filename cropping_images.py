import os

import cv2
import numpy


def cropping_img(filename, coordinates):
    if not isinstance(coordinates, list) or len(coordinates) != 4:
            raise ValueError("Неправильный формат coordinates.\n")
    
    x1, y1, x2, y2 = coordinates

    image = cv2.imread(filename)
    if image is None:
        raise FileNotFoundError(f"Не могу найти изображение: {filename}")
    
    print("Изначальный размер:", image.shape)

    if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
        raise ValueError("Неправильный формат записи координат.")


    cropped_image = image[y1:y2, x1:x2]
    cv2.imshow("cropped", cropped_image)
    key = cv2.waitKey(0)
    print("Новый размер:", cropped_image.shape)
