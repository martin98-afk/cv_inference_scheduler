"""
@author: mading
@license: (C) Copyright: LUCULENT Corporation Limited.
@contact: mading@luculent.net
@file: base64utils.py
@time: 2023/11/3 11:35
@desc: 
"""
import base64
import cv2
import numpy as np

"""对传输的图片进行base64编码"""


def image_to_base64(image_mat: np.array) -> str:
    image = cv2.imencode('.jpg', image_mat)[1]
    image_code = str(base64.b64encode(image), 'utf-8')
    return image_code


###base64图片解码成numpy图
def base64_to_image(imgBase64: str) -> np.array:
    img_data = base64.b64decode(imgBase64)
    bs = np.asarray(bytearray(img_data), dtype='uint8')
    img = cv2.imdecode(bs, cv2.IMREAD_COLOR)
    return img
