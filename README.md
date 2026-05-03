# 🔥 Thermal-Tracking-DETR
### *Full-Stack Object Detection for Long-Wave Infrared (LWIR) Systems*

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/Model-DETR--ResNet50-orange.svg)](https://huggingface.co/docs/transformers/model_doc/detr)

## 📌 Project Overview
This project implements a full-stack **Thermal Tracking System** designed for high-accuracy object localization in zero-light and high-obscuration environments. By leveraging the **DEtection TRansformer (DETR)** architecture, the system provides robust spatial reasoning on thermal heat signatures without the need for traditional non-maximum suppression (NMS).

The model is fine-tuned on the **Teledyne FLIR ADAS Dataset**, specializing in complex urban traffic scenarios involving vehicles and pedestrians.

### Key Features
*   **Thermal-Specific Inference:** Optimized to process 1-channel thermal heat maps rather than standard 3-channel RGB data.
*   **Transformer Architecture:** Implements a ResNet-50 backbone with a Transformer encoder-decoder for end-to-end detection.
*   **Dual-Environment Deployment:** Separates the heavy-duty training logic from the lightweight, responsive web deployment.
*   **Privacy-First Analytics:** Thermal imaging enables behavioral tracking and safety monitoring without capturing sensitive biometric details.

## 🏗️ System Architecture
The project is decoupled into two primary components to ensure modularity:
1.  **Core Model & Training:** Developed using PyTorch and HuggingFace, with weights optimized via GPU-accelerated training loops.
2.  **Web Interface:** A high-performance **FastAPI** backend serving predictions to an interactive **Streamlit** telemetry dashboard.

## 📊 Performance Metrics
*   **Fine-Tuning:** Trained for **15 Epochs** to achieve high-confidence bounding box precision.
*   **Latency:** Optimized for local hardware with an average latency of **0.45 seconds** per frame.
*   **Capacity:** Demonstrated capability to detect and track **35+ concurrent objects** in dense urban environments.

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   Model weights file: `thermal_detr_epoch_15.pth`

### Installation & Setup
1.  **Clone the Web Application Repository:**
    ```bash
    git clone [https://github.com/Hamza237516/Thermal_web_app.git](https://github.com/Hamza237516/Thermal_web_app.git)
    cd Thermal_web_app
    ```
2.  **Initialize Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application
To launch the full-stack system, run both the API and the UI:

1.  **Start the FastAPI Server:**
    ```bash
    uvicorn api:app --reload
    ```
2.  **Launch the Streamlit Dashboard:**
    ```bash
    streamlit run app.py
    ```

## 🛠️ Tech Stack
*   **AI/ML:** PyTorch, HuggingFace Transformers, Timm
*   **Backend:** FastAPI, Uvicorn
*   **Frontend:** Streamlit
*   **Image Processing:** OpenCV, Pillow, NumPy

---
**Developed by [Hamza Mehmood](https://github.com/Hamza237516)**  
*Aspiring MS in Computer Science (AI/ML Focus)*
