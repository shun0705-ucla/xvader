import os
import copy
import numpy as np
import requests
import onnxruntime as ort

SKYSEG_ONNX_URL = "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"
SKYSEG_ONNX_PATH = "skyseg.onnx"


def download_skyseg_onnx(path: str = SKYSEG_ONNX_PATH, url: str = SKYSEG_ONNX_URL):
    """skyseg.onnx が無ければダウンロードする."""
    if os.path.exists(path):
        return
    print(f"[skyseg] Downloading skyseg model to {path} ...")
    resp = requests.get(url, allow_redirects=False)
    resp.raise_for_status()
    if resp.status_code == 302:
        # redirect 先に再リクエスト
        redirect_url = resp.headers["Location"]
        resp = requests.get(redirect_url, stream=True)
        resp.raise_for_status()
    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"[skyseg] Downloaded {path}")


def load_skyseg_session(path: str = SKYSEG_ONNX_PATH) -> ort.InferenceSession:
    """onnxruntime のセッションを作る."""
    download_skyseg_onnx(path)
    sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    return sess


def _run_skyseg_onnx(onnx_session: ort.InferenceSession,
                     image_bgr: np.ndarray,
                     input_size=(320, 320)) -> np.ndarray:
    """
    SkySeg ONNX に 1 枚の BGR 画像を入れて「スコアマップ」を返す。

    戻り値は 0–255 の uint8。値が低いほど sky, 高いほど non-sky。
    """
    import numpy as np
    import cv2

    temp = copy.deepcopy(image_bgr)
    resized = cv2.resize(temp, dsize=input_size)  # (W, H) じゃなくて (width, height)

    x = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    x = (x / 255.0 - mean) / std
    x = x.transpose(2, 0, 1)            # (3, H, W)
    x = x[None].astype("float32")       # (1, 3, H, W)

    in_name = onnx_session.get_inputs()[0].name
    out_name = onnx_session.get_outputs()[0].name
    out = onnx_session.run([out_name], {in_name: x})[0].squeeze()  # (H, W)

    # 0–255 に正規化
    out_min, out_max = float(out.min()), float(out.max())
    out = (out - out_min) / (out_max - out_min + 1e-8)
    out = (out * 255.0).astype("uint8")
    return out


def get_sky_keep_mask(
    rgb: np.ndarray,
    onnx_session: ort.InferenceSession,
    thr: int = 32,
) -> np.ndarray:
    """
    generate mask which keep non-sky pixels from an RGB image.

    Args:
        rgb: (H, W, 3) uint8, RGB
        onnx_session: load_skyseg_session() の戻り値
        thr: threshold for sky segmentation score.

    Returns:
        mask_keep: (H, W) bool, True = non-sky, False = sky
    """
    import cv2
    H, W = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    score = _run_skyseg_onnx(onnx_session, bgr, input_size=(320, 320))  # (h', w')
    score_resized = cv2.resize(score, (W, H), interpolation=cv2.INTER_LINEAR)

    mask_keep = score_resized < thr      # True = non-sky
    return mask_keep