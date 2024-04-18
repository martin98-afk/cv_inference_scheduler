"""
@author: mading
@license: (C) Copyright: LUCULENT Corporation Limited.
@contact: mading@luculent.net
@file: opencv.py
@time: 2023/11/4 10:15
@desc: 
"""
from typing import List, Dict
import cv2.dnn
import numpy as np
from loguru import logger

from cv_inference_scheduler.application.utils.base64utils import image_to_base64
from cv_inference_scheduler.application.utils.time_counter import log_time_cost

_CLASSES: Dict = {}
_COLORS: Dict = {}


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Draws bounding boxes on the input image based on the provided arguments.

    Args:
        img (numpy.ndarray): The input image to draw the bounding box on.
        class_id (int): Class ID of the detected object.
        confidence (float): Confidence score of the detected object.
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
        y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
    """
    label = f'{_CLASSES[class_id]} ({confidence:.2f})'
    color = _COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def object_detection(model: cv2.dnn, original_image: np.ndarray, classes: Dict, colors: np.array) -> List[Dict]:
    """
    Main function to load ONNX model, perform inference, draw bounding boxes, and display the output image.

    Args:
        onnx_model (str): Path to the ONNX model.
        input_image (str): Path to the input image.

    Returns:
        list: List of dictionaries containing detection information such as class_id, class_name, confidence, etc.
    """
    global _COLORS, _CLASSES
    _CLASSES, _COLORS = classes, colors

    @log_time_cost
    def preprocess_func(func_name):
        # Read the input image
        [height, width, _] = original_image.shape

        # Prepare a square image for inference
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image

        # Calculate scale factor
        scale = length / 640

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
        return blob, scale

    @log_time_cost
    def model_forward(func_name):
        model.setInput(blob)

        # Perform inference
        outputs = model.forward()
        return outputs

    @log_time_cost
    def postprocess(func_name):
        boxes = []
        scores = []
        class_ids = []
        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2], outputs[0][i][3]]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.1, nms_threshold=0.3, eta=0.3)

        detections = []
        content_describe = ""
        for cls in set(class_ids):
            n = (np.array(class_ids) == cls).sum()
            content_describe += f"{n} {_CLASSES[class_ids[cls]]}{'s' * (n > 1)} "

        # Iterate through NMS results to draw bounding boxes and labels
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            draw_bounding_box(original_image, class_ids[index], scores[index], round(box[0] * scale),
                              round(box[1] * scale),
                              round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))
            detection = {
                'class_id': int(class_ids[index]),
                'class_name': _CLASSES[class_ids[index]],
                'confidence': float(scores[index]),
                'box': [float(item) for item in box],
                'scale': float(scale)}
            detections.append(detection)

        detections.append({"image": image_to_base64(original_image)})

        return detections, content_describe

    blob, scale = preprocess_func(func_name="PreProcess")
    outputs = model_forward(func_name="Inference")
    # Prepare output array
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]
    detections, content_describe = postprocess(func_name="PostProcess")
    logger.info(f"Detect result: {content_describe}")

    return detections
