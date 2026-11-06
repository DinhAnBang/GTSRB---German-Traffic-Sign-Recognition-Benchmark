"""Streamlit UI for testing the GTSRB ONNX traffic-sign classifier."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

from inference import (
    DEFAULT_IMAGE_SIZE,
    iter_image_files,
    load_labels,
    predict_image,
    split_into_quadrants,
)


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "GTSRB.onnx"
DEFAULT_LABELS_PATH = ROOT_DIR / "labels.json"


st.set_page_config(
    page_title="GTSRB Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_session(model_path: str) -> ort.InferenceSession:
    """Cache the ONNX session so the model is not reloaded on every click."""

    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


@st.cache_data(show_spinner=False)
def get_labels(labels_path: str) -> dict[int, dict[str, str]]:
    return load_labels(labels_path)


def class_title(class_id: int, labels: dict[int, dict[str, str]]) -> str:
    label = labels.get(class_id, {"en": f"Class {class_id}", "vi": f"Lớp {class_id}"})
    return f"{label['vi']} / {label['en']}"


def prediction_table(predictions: list[dict], labels: dict[int, dict[str, str]]) -> pd.DataFrame:
    rows = []
    for rank, item in enumerate(predictions, start=1):
        row = {
            "Hạng": rank,
            "Lớp": f"{item['class_id']:02d}",
            "Biển báo": class_title(item["class_id"], labels),
            "Độ tin cậy": f"{item['confidence']:.2%}",
        }
        if "quadrant" in item:
            row = {"Vùng": item["quadrant"], **row}
        rows.append(row)
    return pd.DataFrame(rows)


def draw_quadrant_boxes(
    image: Image.Image,
    quadrant_results: list[dict],
    confidence_threshold: float,
) -> Image.Image:
    """Draw the four preprocessing boxes and each region's best class."""

    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    colors = {
        "Q1": (255, 72, 72),
        "Q2": (67, 160, 255),
        "Q3": (58, 190, 115),
        "Q4": (255, 177, 66),
    }
    line_width = max(3, round(min(annotated.size) / 180))

    for result in quadrant_results:
        name = result["name"]
        x1, y1, x2, y2 = result["box"]
        best = result["predictions"][0] if result.get("predictions") else None
        if best is None:
            label = f"{name} | no result"
        elif best["confidence"] >= confidence_threshold:
            label = f"{name} | {best['class_id']:02d} {best['confidence']:.1%}"
        else:
            label = f"{name} | below {confidence_threshold:.0%}"

        color = colors.get(name, (255, 255, 255))
        draw.rectangle((x1, y1, max(x1, x2 - 1), max(y1, y2 - 1)), outline=color, width=line_width)
        text_box = draw.textbbox((0, 0), label)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_bottom = min(y1 + text_height + 8, annotated.height)
        draw.rectangle((x1, y1, min(x1 + text_width + 10, annotated.width), label_bottom), fill=color)
        draw.text((x1 + 5, y1 + 3), label, fill=(0, 0, 0))

    return annotated


def render_sidebar() -> tuple[str, str, float, int]:
    st.sidebar.title("🚦 GTSRB Recognizer")
    st.sidebar.caption("Công cụ nhận diện biển báo bằng ONNX Runtime")

    model_path = st.sidebar.text_input("Đường dẫn model", str(DEFAULT_MODEL_PATH))
    labels_path = st.sidebar.text_input("Đường dẫn labels", str(DEFAULT_LABELS_PATH))
    confidence_threshold = st.sidebar.slider(
        "Confidence tối thiểu để nhận diện",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.01,
        format="%.0f%%",
        help="Chỉ hiển thị các lớp có confidence lớn hơn hoặc bằng ngưỡng này.",
    )
    top_k = st.sidebar.slider("Số vùng cắt hiển thị", 1, 4, 4)

    st.sidebar.divider()
    st.sidebar.caption("Input model: 3 × 128 × 128 · Output: 43 classes")
    st.sidebar.caption("Nếu không có kết quả, hãy kéo confidence xuống thấp hơn.")
    return model_path, labels_path, confidence_threshold, top_k


def render_prediction_tab(
    session: ort.InferenceSession,
    labels: dict[int, dict[str, str]],
    confidence_threshold: float,
    top_k: int,
) -> None:
    st.subheader("Nhận diện ảnh mới")
    st.write(
        "Tải lên ảnh có biển báo. Ảnh sẽ được chia thành 4 vùng, giữ tỷ lệ, padding "
        "và đưa từng vùng về kích thước 128 × 128."
    )

    uploaded_file = st.file_uploader(
        "Chọn ảnh biển báo",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Hãy chọn một ảnh để bắt đầu nhận diện.")
        return

    try:
        image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
    except Exception as error:  # pragma: no cover - UI error path
        st.error(f"Không thể đọc ảnh: {error}")
        return

    upload_key = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.get("prediction_upload_key") != upload_key:
        st.session_state["prediction_upload_key"] = upload_key
        st.session_state.pop("last_quadrant_predictions", None)
        st.session_state.pop("last_filename", None)

    try:
        quadrants = split_into_quadrants(image)
    except ValueError as error:
        st.error(str(error))
        return

    action_left, action_right = st.columns([1.1, 1], gap="large")
    with action_right:
        if st.button("🔎 Nhận diện 4 vùng", type="primary", use_container_width=True):
            try:
                quadrant_predictions = []
                for quadrant in quadrants:
                    quadrant_predictions.append(
                        {
                            "name": quadrant["name"],
                            "box": quadrant["box"],
                            "predictions": predict_image(
                                session,
                                quadrant["image"],
                                size=DEFAULT_IMAGE_SIZE,
                                top_k=43,
                            ),
                        }
                    )
                st.session_state["last_quadrant_predictions"] = quadrant_predictions
                st.session_state["last_filename"] = uploaded_file.name
            except Exception as error:  # pragma: no cover - UI error path
                st.error(f"Không thể chạy model: {error}")

    quadrant_results = st.session_state.get("last_quadrant_predictions")
    annotation_results = quadrant_results or [
        {"name": quadrant["name"], "box": quadrant["box"], "predictions": []}
        for quadrant in quadrants
    ]
    annotated_image = draw_quadrant_boxes(image, annotation_results, confidence_threshold)

    image_column, result_column = st.columns([1.1, 1], gap="large")
    with image_column:
        st.image(
            annotated_image,
            caption="Ảnh đầu vào với 4 khung cắt Q1–Q4",
            use_container_width=True,
        )
        st.caption(
            "Mỗi khung là một vùng được đưa vào model. Nhãn trong khung xuất hiện sau khi bấm Nhận diện."
        )

    with result_column:
        if not quadrant_results:
            st.info("Bấm Nhận diện 4 vùng để xem nhãn và confidence trên từng khung.")
        else:
            recognized_predictions = []
            for quadrant_result in quadrant_results:
                best = quadrant_result["predictions"][0]
                if best["confidence"] >= confidence_threshold:
                    recognized_predictions.append(
                        {
                            **best,
                            "quadrant": quadrant_result["name"],
                        }
                    )
            recognized_predictions.sort(key=lambda item: item["confidence"], reverse=True)
            recognized_predictions = recognized_predictions[:top_k]

            if not recognized_predictions:
                best_candidate = max(
                    (result["predictions"][0] for result in quadrant_results),
                    key=lambda item: item["confidence"],
                )
                st.warning(
                    f"Chưa có vùng nào đạt confidence {confidence_threshold:.0%}. "
                    f"Điểm cao nhất là lớp {best_candidate['class_id']:02d} "
                    f"với {best_candidate['confidence']:.2%}. "
                    "Hãy kéo ngưỡng confidence xuống hoặc chọn ảnh rõ hơn."
                )
            else:
                best = recognized_predictions[0]
                st.metric(
                    "Dự đoán chính",
                    f"{best['quadrant']} · {best['class_id']:02d} · {class_title(best['class_id'], labels)}",
                    f"{best['confidence']:.2%}",
                )
                st.success(
                    f"Có {len(recognized_predictions)} vùng đạt confidence từ "
                    f"{confidence_threshold:.0%} trở lên."
                )
                st.dataframe(
                    prediction_table(recognized_predictions, labels),
                    hide_index=True,
                    use_container_width=True,
                )

                export_payload = {
                    "filename": st.session_state.get("last_filename", uploaded_file.name),
                    "confidence_threshold": confidence_threshold,
                    "quadrants": [
                        {
                            "name": result["name"],
                            "box": result["box"],
                            "best_prediction": result["predictions"][0],
                        }
                        for result in quadrant_results
                    ],
                    "predictions": recognized_predictions,
                }
                st.download_button(
                    "Tải kết quả JSON",
                    data=json.dumps(export_payload, ensure_ascii=False, indent=2),
                    file_name="prediction.json",
                    mime="application/json",
                    use_container_width=True,
                )

            with st.expander("Kiểm tra đúng / sai", expanded=True):
                st.caption(
                    "Ảnh mới không có nhãn thật nên confidence không thể chứng minh đúng tuyệt đối. "
                    "Chọn nhãn thật nếu bạn biết để kiểm tra trực tiếp."
                )
                truth_options = ["-- Chưa biết --"] + [
                    f"{class_id:02d} · {class_title(class_id, labels)}"
                    for class_id in sorted(labels)
                ]
                for result in quadrant_results:
                    predicted = result["predictions"][0]
                    selected_truth = st.selectbox(
                        f"Nhãn thật của {result['name']}",
                        truth_options,
                        key=f"truth_{upload_key}_{result['name']}",
                    )
                    if selected_truth != "-- Chưa biết --":
                        expected_class = int(selected_truth.split(" · ", 1)[0])
                        if expected_class == predicted["class_id"]:
                            st.success(
                                f"{result['name']}: ĐÚNG — model dự đoán "
                                f"{predicted['class_id']:02d} với {predicted['confidence']:.2%}."
                            )
                        else:
                            st.error(
                                f"{result['name']}: SAI — nhãn thật {expected_class:02d}, "
                                f"model dự đoán {predicted['class_id']:02d}."
                            )

    with st.expander("Xem ảnh sau tiền xử lý", expanded=True):
        st.caption(
            "Các khung màu là 4 vùng cắt cố định trước khi đưa vào model. "
            "Đây chưa phải bounding box tự động của model classification."
        )
        st.image(annotated_image, caption="Ảnh với khung và nhãn dự đoán", use_container_width=True)

        crop_columns = st.columns(4)
        for column, quadrant in zip(crop_columns, quadrants):
            with column:
                st.image(quadrant["image"], caption=quadrant["name"], use_container_width=True)
                result = next(
                    (item for item in annotation_results if item["name"] == quadrant["name"]),
                    None,
                )
                if result and result.get("predictions"):
                    best = result["predictions"][0]
                    st.caption(
                        f"{class_title(best['class_id'], labels)} · {best['confidence']:.2%}"
                    )


def render_validation_tab(
    session: ort.InferenceSession,
    labels: dict[int, dict[str, str]],
) -> None:
    st.subheader("Đánh giá model trên một dataset")
    st.write("Dataset cần có dạng thư mục số lớp, ví dụ `dataset/0`, `dataset/1`, …, `dataset/42`.")
    dataset_path = st.text_input("Đường dẫn dataset validation", str(ROOT_DIR / "dataset"))

    if not st.button("▶ Chạy validation", type="primary"):
        st.info("Chọn thư mục dataset rồi bấm Chạy validation.")
        return

    try:
        samples = list(iter_image_files(dataset_path))
    except FileNotFoundError as error:
        st.error(str(error))
        return

    if not samples:
        st.warning("Không tìm thấy ảnh trong các thư mục lớp dạng số.")
        return

    progress = st.progress(0, text="Đang đánh giá...")
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    correct = 0
    errors: list[dict[str, str]] = []

    for index, (expected_class, image_path) in enumerate(samples, start=1):
        try:
            image = Image.open(image_path).convert("RGB")
            prediction = predict_image(session, image, top_k=1)[0]
            predicted_class = prediction["class_id"]
            if expected_class < confusion.shape[0] and predicted_class < confusion.shape[1]:
                confusion[expected_class, predicted_class] += 1
            if predicted_class == expected_class:
                correct += 1
            elif len(errors) < 100:
                errors.append(
                    {
                        "file": str(image_path),
                        "expected": f"{expected_class:02d} · {class_title(expected_class, labels)}",
                        "predicted": f"{predicted_class:02d} · {class_title(predicted_class, labels)}",
                        "confidence": f"{prediction['confidence']:.2%}",
                    }
                )
        except Exception as error:
            if len(errors) < 100:
                errors.append({"file": str(image_path), "expected": "Lỗi đọc ảnh", "predicted": str(error)})

        progress.progress(index / len(samples), text=f"Đang đánh giá {index}/{len(samples)}")

    accuracy = correct / len(samples)
    st.metric("Accuracy", f"{accuracy:.2%}", f"{correct}/{len(samples)} ảnh đúng")

    st.subheader("Confusion matrix")
    matrix_labels = [f"{class_id:02d}" for class_id in range(len(labels))]
    st.dataframe(
        pd.DataFrame(confusion, index=matrix_labels, columns=matrix_labels),
        use_container_width=True,
    )

    if errors:
        st.subheader("Một số mẫu dự đoán sai hoặc lỗi")
        st.dataframe(pd.DataFrame(errors), hide_index=True, use_container_width=True)


def render_model_info(session: ort.InferenceSession, model_path: str) -> None:
    st.subheader("Thông tin model")
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    info = {
        "Model": model_path,
        "Provider": ", ".join(session.get_providers()),
        "Input name": input_meta.name,
        "Input shape": str(input_meta.shape),
        "Input type": input_meta.type,
        "Output name": output_meta.name,
        "Output shape": str(output_meta.shape),
        "Output type": output_meta.type,
    }
    st.dataframe(pd.DataFrame([info]), hide_index=True, use_container_width=True)
    st.info(
        "File ONNX dùng để suy luận/validation. Muốn train lại cần dataset và checkpoint huấn luyện "
        "(ví dụ .pt); không thể train trực tiếp từ GTSRB.onnx."
    )


def main() -> None:
    model_path, labels_path, confidence_threshold, top_k = render_sidebar()
    st.title("🚦 GTSRB Traffic Sign Recognition")
    st.caption("Nhận diện biển báo giao thông bằng model ONNX")

    model_file = Path(model_path)
    labels_file = Path(labels_path)
    if not model_file.is_file():
        st.error(f"Không tìm thấy model: {model_file}")
        st.stop()
    if not labels_file.is_file():
        st.error(f"Không tìm thấy file labels: {labels_file}")
        st.stop()

    try:
        session = get_session(str(model_file.resolve()))
        labels = get_labels(str(labels_file.resolve()))
    except Exception as error:  # pragma: no cover - UI error path
        st.error(f"Không thể load model hoặc labels: {error}")
        st.stop()

    prediction_tab, validation_tab, info_tab = st.tabs(["Nhận diện", "Validation", "Thông tin model"])
    with prediction_tab:
        render_prediction_tab(session, labels, confidence_threshold, top_k)
    with validation_tab:
        render_validation_tab(session, labels)
    with info_tab:
        render_model_info(session, str(model_file.resolve()))


if __name__ == "__main__":
    main()

