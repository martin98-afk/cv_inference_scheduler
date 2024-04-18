import onnxruntime as ort
from loguru import logger


def load_onnx_model(onnx_path: str, device="cpu") -> ort.InferenceSession:
    options = {} if device == "cpu" else {"device_id": device}
    try:
        session = ort.InferenceSession(onnx_path)
        session.set_providers(["CUDAExecutionProvider"]
            if device != "cpu" and ort.get_device() == "GPU"
            else ["CPUExecutionProvider"], [options])

    except Exception as e:
        import traceback
        logger.error(f"load onnx model:  {onnx_path} failed ! error info: \n{traceback.format_exc()}")
        raise e

    return session


def load_pytorch_model(pt_path: str, device="cpu") -> ort.InferenceSession:
    raise  NotImplementedError


# def load_pytorch_model(pt_path: str, device="cpu") -> ort.InferenceSession:
#     from application.utils.model_converter import pt2onnx
#     pt2onnx(pt_path, save_dir=pt_path.replace(".pt", ".onnx"))
#     return load_onnx_model(pt_path.replace(".pt", ".onnx"), device)


def load_tensorflow_model(tf_path: str):
    pass


if __name__ == '__main__':
    pass
