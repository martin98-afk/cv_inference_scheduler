import os
import argparse

import cv2
import numpy as np
import onnxruntime as ort

from application.utils.yaml_loader import yaml_load


class YOLOv8Pose:
    """YOLOv8 segmentation model."""

    def __init__(self, onnx_model: ort.InferenceSession, data_config: str):
        """
        Initialization.

        Args:
            onnx_model (str): Path to the ONNX model.
        """

        # Build Ort session
        self.session = onnx_model

        # Numpy dtype: support both FP32 and FP16 onnx model
        self.ndtype = np.half if self.session.get_inputs()[0].type == "tensor(float16)" else np.single

        # Get model width and height(YOLOv8-seg only has one input)
        self.model_height, self.model_width = [x.shape for x in self.session.get_inputs()][0][-2:]

        # Load COCO class names
        self.classes = yaml_load(data_config)["names"]

    def __call__(self, im0, conf_threshold=0.4, iou_threshold=0.45):
        """
        The whole pipeline: pre-process -> inference -> post-process.

        Args:
            im0 (Numpy.ndarray): original input image.
            conf_threshold (float): confidence threshold for filtering predictions.
            iou_threshold (float): iou threshold for NMS.
            nm (int): the number of masks.

        Returns:
            boxes (List): list of bounding boxes.
            segments (List): list of segments.
            masks (np.ndarray): [N, H, W], output masks.
        """

        # Pre-process
        im, ratio, (pad_w, pad_h) = self.preprocess(im0)

        # Ort inference
        preds = self.session.run(None, {self.session.get_inputs()[0].name: im})

        # Post-process
        bboxes, keypoint1, keypoint2, obj_class, scores = self.postprocess(
            preds,
            im0=im0,
            ratio=ratio,
            pad_w=pad_w,
            pad_h=pad_h,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            class_num=len(self.classes),
        )
        return bboxes, keypoint1, keypoint2, obj_class, scores

    def preprocess(self, img):
        """
        Pre-processes the input image.

        Args:
            img (Numpy.ndarray): image about to be processed.

        Returns:
            img_process (Numpy.ndarray): image preprocessed for inference.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
        """

        # Resize and pad input image using letterbox() (Borrowed from Ultralytics)
        shape = img.shape[:2]  # original image shape
        new_shape = (self.model_height, self.model_width)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Transforms: HWC to CHW -> BGR to RGB -> div(255) -> contiguous -> add axis(optional)
        img = np.ascontiguousarray(np.einsum("HWC->CHW", img)[::-1], dtype=self.ndtype) / 255.0
        img_process = img[None] if len(img.shape) == 3 else img
        return img_process, ratio, (pad_w, pad_h)

    def postprocess(self, preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold, class_num):
        """
        Post-process the prediction.

        Args:
            preds (Numpy.ndarray): predictions come from ort.session.run().
            im0 (Numpy.ndarray): [h, w, c] original input image.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
            conf_threshold (float): conf threshold.
            iou_threshold (float): iou threshold.
            nm (int): the number of masks.

        Returns:
            boxes (List): list of bounding boxes.
            segments (List): list of segments.
            masks (np.ndarray): [N, H, W], output masks.
        """
        x = preds[0]  # Two outputs: predictions and protos

        # Transpose the first output: (Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
        x = np.einsum("bcn->bnc", x)

        # Predictions filtering by conf-threshold
        x = np.squeeze(x)

        # NMS filtering
        result = []
        obj_class = []
        scores = []
        for i in range(class_num):
            nms_index = cv2.dnn.NMSBoxes(x[:, :4], x[:, 4 + i], conf_threshold, iou_threshold)
            result.append(np.squeeze(x[list(nms_index)]))
            if result[i].ndim == 1:
                obj_class.extend([i])
                scores.append(result[-1][4 + i])
            else:
                obj_class.extend([i] * result[i].shape[0])
                scores.extend(result[-1][:, 4 + i].tolist())

        x = result
        # Decode and return
        if len(x) > 0:
            bbox = []
            keypoints1 = []
            keypoints2 = []
            for i in range(len(x)):
                # Bounding boxes format change: cxcywh -> xyxy
                x[i][..., [0, 1]] -= x[i][..., [2, 3]] / 2
                x[i][..., [2, 3]] += x[i][..., [0, 1]]

                # Rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
                x[i][..., :4] -= [pad_w, pad_h, pad_w, pad_h]
                x[i][..., :4] /= min(ratio)

                # Bounding boxes boundary clamp
                x[i][..., [0, 2]] = x[i][..., [0, 2]].clip(0, im0.shape[1])
                x[i][..., [1, 3]] = x[i][..., [1, 3]].clip(0, im0.shape[0])
                x[i] = x[i].reshape(1, -1) if x[i].ndim == 1 else x[i]
                bbox.extend(x[i][..., :4].tolist())
                keypoints1.extend(((x[i][..., 6::3] - pad_w) / min(ratio)).tolist())
                keypoints2.extend(((x[i][..., 7::3] - pad_w) / min(ratio)).tolist())
            # Masks -> Segments(contours)
            return bbox, keypoints1, keypoints2, obj_class, scores
        else:
            return [], [], [], []

    def draw_and_visualize(self, im, bboxes, keypoints1, keypoints2, class_obj, vis=False, save=True):
        """
        Draw and visualize results.

        Args:
            im (np.ndarray): original image, shape [h, w, c].
            bboxes (numpy.ndarray): [n, 4], n is number of bboxes.
            segments (List): list of segment masks.
            vis (bool): imshow using OpenCV.
            save (bool): save image annotated.

        Returns:
            None
        """

        # Draw rectangles and polygons
        im_canvas = im.copy()
        for cls_, box, kp_x, kp_y in zip(class_obj, bboxes, keypoints1, keypoints2):

            # draw bbox rectangle
            cv2.rectangle(
                im,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                self.color_palette(int(cls_), bgr=True),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                im,
                f"{self.classes[cls_]}",
                (int(box[0]), int(box[1] - 9)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.color_palette(int(cls_), bgr=True),
                2,
                cv2.LINE_AA,
            )
            for x, y in zip(kp_x, kp_y):
                cv2.circle(im, (int(x), int(y)), 10, self.color_palette(int(cls_), bgr=True), -1)

        # Mix image
        im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)

        # Show image
        if vis:
            cv2.imshow("demo", im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Save image
        if save:
            cv2.imwrite("demo.jpg", im)


if __name__ == "__main__":

    # Create an argument parser to handle command-line arguments
    pt_path = "/home/mading/object_detection/cv_inference_scheduler/weights/lifting_hook_best.pt"
    from ultralytics import YOLO

    if not os.path.exists(pt_path.replace(".pt", ".onnx")):
        model = YOLO(pt_path)
        model.export(format="onnx", save_dir=pt_path.replace(".pt", ".onnx"))

    model = ort.InferenceSession(
        pt_path.replace(".pt", ".onnx"),
        providers=["CPUExecutionProvider"],
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=str("1.png"), help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.9, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    args = parser.parse_args()

    # Build model
    model = YOLOv8Pose(model, "/home/mading/object_detection/cv_inference_scheduler/yml/lifting_hook.yaml")

    # Read image by OpenCV
    img = cv2.imread(args.source)

    # Inference
    bbox, keypoint1, keypoint2, obj_class, _ = model(img, conf_threshold=args.conf, iou_threshold=args.iou)

    # Draw bboxes and polygons
    if len(bbox) > 0:
        model.draw_and_visualize(img, bbox, keypoint1, keypoint2, obj_class, vis=False, save=True)
