# Ultralytics YOLO 🚀, AGPL-3.0 license
import argparse
import os
import cv2
import numpy as np
import onnxruntime as ort

from application.utils.yaml_loader import yaml_load


class YOLOv8:
    """YOLOv8 object detection model class for handling inference and visualization."""

    def __init__(self, onnx_model: ort.InferenceSession, data_config: str):
        """
        Initializes an instance of the YOLOv8 class.

        Args:
            onnx_model: ONNX model.
        """
        self.onnx_model = onnx_model

        # Load the class names from the COCO dataset
        self.classes = yaml_load(data_config)["names"]

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, img):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """

        # Get the height and width of the input image
        self.img_height, self.img_width = img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data

    def postprocess(self, output, confidence_thres, iou_thres):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            output (numpy.ndarray): The output of the model.
        """

        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)
        indices = list(np.sort(indices).astype(np.int))
        # Iterate over the selected indices after non-maximum suppression
            # Draw the detection on the input image
            # self.draw_detections(input_image, box, score, class_id)

        # Return the modified input image
        return [boxes[i] for i in indices], [scores[i] for i in indices], [class_ids[i] for i in indices]
        # return boxes[indices],

    def __call__(self, input_image, conf_threshold, iou_threshold):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
        # Create an inference session using the ONNX model and specify execution providers
        session = self.onnx_model

        # Get the model inputs
        model_inputs = session.get_inputs()

        # Store the shape of the input for later use
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        # Preprocess the image data
        img_data = self.preprocess(input_image)

        # Run inference using the preprocessed image data
        outputs = session.run(None, {model_inputs[0].name: img_data})

        # Perform post-processing on the outputs to obtain output image.
        return self.postprocess(outputs, conf_threshold, iou_threshold)  # output image


if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    pt_path = "/home/mading/object_detection/cv_inference_scheduler/weights/yolov8s.pt"
    from ultralytics import YOLO

    if not os.path.exists(pt_path.replace(".pt", ".onnx")):
        model = YOLO(pt_path)
        model.export(format="onnx", save_dir=pt_path.replace(".pt", ".onnx"))

    model = ort.InferenceSession(
        pt_path.replace(".pt", ".onnx"),
        providers=["CPUExecutionProvider"],
    )
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", type=str, default=str("img.png"), help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    args = parser.parse_args()

    # Build model
    model = YOLOv8(model, data_config="/home/mading/object_detection/cv_inference_scheduler/yml/coco.yaml")

    # Read image by OpenCV
    img = cv2.imread(args.source)

    # Inference
    box, score, class_id = model(img, conf_threshold=args.conf, iou_threshold=args.iou)

    for b, s, c in zip(box, score, class_id):
        # Draw bboxes
        model.draw_detections(img, b, s, c)

    cv2.imwrite("demo.jpg", img)
    # # Draw bboxes and polygons
    # if len(boxes) > 0:
    #     model.draw_and_visualize(img, boxes, segments, vis=False, save=True)
