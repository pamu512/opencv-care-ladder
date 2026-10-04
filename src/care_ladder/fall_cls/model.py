"""Tiny MobileNet-class CNN (depthwise-separable), numpy train, ONNX export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from care_ladder.fall_cls.data import Sample, preprocess_bgr


@dataclass
class TrainedModel:
    conv1_w: np.ndarray
    conv1_b: np.ndarray
    dw_w: np.ndarray
    dw_b: np.ndarray
    pw_w: np.ndarray
    pw_b: np.ndarray
    fc_w: np.ndarray
    fc_b: np.ndarray
    size: int = 32
    class_names: tuple[str, ...] = ("no_fall", "fall")


@dataclass(frozen=True)
class Metrics:
    sensitivity: float
    specificity: float
    accuracy: float
    confusion: tuple[int, int, int, int]  # tn, fp, fn, tp
    n: int
    note: str = ""


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _conv2d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, *, pad: int = 1, groups: int = 1) -> np.ndarray:
    n, c, h, w = x.shape
    out_c, in_c, k, _ = weight.shape
    if groups == 1:
        assert in_c == c
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((n, out_c, h, w), dtype=np.float32)
    if groups == 1:
        for i in range(k):
            for j in range(k):
                patch = xp[:, :, i : i + h, j : j + w]
                out += np.einsum("nchw,oc->nohw", patch, weight[:, :, i, j], optimize=True)
    else:
        if out_c != c or groups != c:
            raise ValueError("depthwise conv expects groups == in_channels == out_channels")
        for ch in range(c):
            for i in range(k):
                for j in range(k):
                    out[:, ch] += xp[:, ch, i : i + h, j : j + w] * float(weight[ch, 0, i, j])
    out += bias.reshape(1, -1, 1, 1)
    return out


def _maxpool2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, c, h, w = x.shape
    hs, ws = h // 2 * 2, w // 2 * 2
    x = x[:, :, :hs, :ws]
    shaped = x.reshape(n, c, hs // 2, 2, ws // 2, 2)
    flat = shaped.transpose(0, 1, 2, 4, 3, 5).reshape(n, c, hs // 2, ws // 2, 4)
    idx = flat.argmax(axis=-1)
    pooled = flat.max(axis=-1)
    return pooled, idx


def _unpool2(grad: np.ndarray, idx: np.ndarray) -> np.ndarray:
    n, c, h, w = grad.shape
    out = np.zeros((n, c, h, w, 4), dtype=np.float32)
    n_i, c_i, h_i, w_i = np.indices(idx.shape)
    out[n_i, c_i, h_i, w_i, idx] = grad
    out = out.reshape(n, c, h, w, 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h * 2, w * 2)
    return out


def _forward(x: np.ndarray, model: TrainedModel, *, cache: bool = False):
    c1 = _conv2d(x, model.conv1_w, model.conv1_b, pad=1, groups=1)
    r1 = _relu(c1)
    p1, idx1 = _maxpool2(r1)
    dw = _conv2d(p1, model.dw_w, model.dw_b, pad=1, groups=p1.shape[1])
    pw = _conv2d(dw, model.pw_w, model.pw_b, pad=0, groups=1)
    r2 = _relu(pw)
    gap = r2.mean(axis=(2, 3))
    logits = gap @ model.fc_w.T + model.fc_b
    if not cache:
        return logits
    return logits, {
        "x": x,
        "c1": c1,
        "r1": r1,
        "p1": p1,
        "idx1": idx1,
        "dw": dw,
        "pw": pw,
        "r2": r2,
        "gap": gap,
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _load_batch(samples: Sequence[Sample], size: int) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    xs = []
    ys = []
    for sample in samples:
        image = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"unreadable image: {sample.path}")
        xs.append(preprocess_bgr(image, size=size))
        ys.append(int(sample.label))
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def train_classifier(
    samples: Sequence[Sample],
    *,
    seed: int = 47,
    epochs: int = 12,
    lr: float = 0.12,
    size: int = 32,
) -> TrainedModel:
    if len(samples) < 4:
        raise ValueError("need at least 4 labeled frames to train")
    rng = np.random.default_rng(seed)
    model = TrainedModel(
        conv1_w=rng.normal(0, 0.12, (8, 3, 3, 3)).astype(np.float32),
        conv1_b=np.zeros(8, dtype=np.float32),
        dw_w=rng.normal(0, 0.12, (8, 1, 3, 3)).astype(np.float32),
        dw_b=np.zeros(8, dtype=np.float32),
        pw_w=rng.normal(0, 0.12, (16, 8, 1, 1)).astype(np.float32),
        pw_b=np.zeros(16, dtype=np.float32),
        fc_w=rng.normal(0, 0.12, (2, 16)).astype(np.float32),
        fc_b=np.zeros(2, dtype=np.float32),
        size=size,
    )
    x, y = _load_batch(samples, size)
    n = len(samples)
    for _ in range(epochs):
        order = rng.permutation(n)
        xb, yb = x[order], y[order]
        logits, cache = _forward(xb, model, cache=True)
        probs = _softmax(logits)
        onehot = np.zeros_like(probs)
        onehot[np.arange(n), yb] = 1.0
        dlogits = (probs - onehot) / n
        dgap = dlogits @ model.fc_w
        dfc_w = dlogits.T @ cache["gap"]
        dfc_b = dlogits.sum(axis=0)
        n_, c_, h_, w_ = cache["r2"].shape
        dr2 = np.broadcast_to((dgap / (h_ * w_))[:, :, None, None], cache["r2"].shape).copy()
        dpw = dr2 * (cache["r2"] > 0)
        # pointwise 1x1 conv backward
        dpw_w = np.einsum("nohw,nihw->oi", dpw, cache["dw"], optimize=True).reshape(16, 8, 1, 1)
        dpw_b = dpw.sum(axis=(0, 2, 3))
        ddw = np.einsum("nohw,oi->nihw", dpw, model.pw_w[:, :, 0, 0], optimize=True)
        # depthwise conv backward (pad 1)
        pad = 1
        p1p = np.pad(cache["p1"], ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        ddw_w = np.zeros_like(model.dw_w)
        ddw_b = ddw.sum(axis=(0, 2, 3))
        dp1 = np.zeros_like(cache["p1"])
        for ch in range(8):
            for i in range(3):
                for j in range(3):
                    patch = p1p[:, ch, i : i + h_, j : j + w_]
                    ddw_w[ch, 0, i, j] = float((patch * ddw[:, ch]).sum())
                    dp1[:, ch] += ddw[:, ch] * float(model.dw_w[ch, 0, i, j])
        dr1 = _unpool2(dp1, cache["idx1"])
        dc1 = dr1 * (cache["r1"] > 0)
        h1, w1 = dc1.shape[2], dc1.shape[3]
        xp = np.pad(cache["x"], ((0, 0), (0, 0), (1, 1), (1, 1)))
        dc1_w = np.zeros_like(model.conv1_w)
        for i in range(3):
            for j in range(3):
                patch = xp[:, :, i : i + h1, j : j + w1]
                dc1_w[:, :, i, j] = np.einsum("nohw,nchw->oc", dc1, patch, optimize=True)
        dc1_b = dc1.sum(axis=(0, 2, 3))
        model.fc_w -= lr * dfc_w.astype(np.float32)
        model.fc_b -= lr * dfc_b.astype(np.float32)
        model.pw_w -= lr * dpw_w.astype(np.float32)
        model.pw_b -= lr * dpw_b.astype(np.float32)
        model.dw_w -= lr * ddw_w.astype(np.float32)
        model.dw_b -= lr * ddw_b.astype(np.float32)
        model.conv1_w -= lr * dc1_w.astype(np.float32)
        model.conv1_b -= lr * dc1_b.astype(np.float32)
    return model


def evaluate(model: TrainedModel, samples: Sequence[Sample]) -> Metrics:
    if not samples:
        return Metrics(0.0, 0.0, 0.0, (0, 0, 0, 0), 0, note="empty eval set")
    x, y = _load_batch(samples, model.size)
    probs = _softmax(_forward(x, model))
    pred = probs.argmax(axis=1)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    pos = tp + fn
    neg = tn + fp
    return Metrics(
        sensitivity=(tp / pos) if pos else 0.0,
        specificity=(tn / neg) if neg else 0.0,
        accuracy=(tp + tn) / max(len(samples), 1),
        confusion=(tn, fp, fn, tp),
        n=len(samples),
    )


def export_onnx(model: TrainedModel, path: Path) -> Path:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    size = model.size
    images = helper.make_tensor_value_info("images", TensorProto.FLOAT, ["N", 3, size, size])
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, ["N", 2])
    initializers = [
        numpy_helper.from_array(model.conv1_w, "conv1_w"),
        numpy_helper.from_array(model.conv1_b, "conv1_b"),
        numpy_helper.from_array(model.dw_w, "dw_w"),
        numpy_helper.from_array(model.dw_b, "dw_b"),
        numpy_helper.from_array(model.pw_w, "pw_w"),
        numpy_helper.from_array(model.pw_b, "pw_b"),
        numpy_helper.from_array(model.fc_w, "fc_w"),
        numpy_helper.from_array(model.fc_b, "fc_b"),
    ]
    nodes = [
        helper.make_node("Conv", ["images", "conv1_w", "conv1_b"], ["c1"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c1"], ["r1"]),
        helper.make_node("MaxPool", ["r1"], ["p1"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node(
            "Conv",
            ["p1", "dw_w", "dw_b"],
            ["dw"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            group=8,
        ),
        helper.make_node("Conv", ["dw", "pw_w", "pw_b"], ["pw"], kernel_shape=[1, 1]),
        helper.make_node("Relu", ["pw"], ["r2"]),
        helper.make_node("GlobalAveragePool", ["r2"], ["gap4"]),
        helper.make_node("Flatten", ["gap4"], ["gap"], axis=1),
        helper.make_node("Gemm", ["gap", "fc_w", "fc_b"], ["logits"], transB=1),
        helper.make_node("Softmax", ["logits"], ["scores"], axis=1),
    ]
    graph = helper.make_graph(nodes, "fall_cls_v1", [images], [scores], initializers)
    proto = helper.make_model(
        graph,
        producer_name="opencv-care-ladder",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    proto.ir_version = 8
    onnx.checker.check_model(proto)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(proto, str(path))
    return path


def load_opencv_net(path: Path):
    import cv2

    net = cv2.dnn.readNetFromONNX(str(path))
    if net.empty():
        raise RuntimeError(f"OpenCV DNN failed to load {path}")
    return net


def predict_opencv(net, image_bgr: np.ndarray, size: int = 32) -> tuple[int, np.ndarray]:
    blob = preprocess_bgr(image_bgr, size=size)[None]
    net.setInput(blob)
    out = net.forward()
    probs = np.asarray(out, dtype=np.float32).reshape(-1)
    if probs.size != 2:
        raise RuntimeError(f"expected 2-class scores, got {probs.shape}")
    s = float(probs.sum())
    if s <= 0:
        raise RuntimeError("onnx scores summed to 0")
    probs = probs / s
    return int(probs.argmax()), probs
