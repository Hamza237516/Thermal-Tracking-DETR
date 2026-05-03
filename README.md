streamlit run app.py
    ```

## 🛠️ Tech Stack
*   **Core AI:** PyTorch, HuggingFace Transformers, Timm.
*   **Web Frameworks:** FastAPI, StreamlitA professional `README.md` acts as the front door to your project. Since you are developing this for your Master's applications, it needs to highlight both your engineering skills and your theoretical understanding of AI.

Below is a high-performance template tailored for your **Thermal-Tracking-DETR** project.

---

# 🔥 Thermal-Tracking-DETR
### *Autonomous Object Detection for Long-Wave Infrared (LWIR) Systems*

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/Model-DETR--ResNet50-orange.svg)](https://huggingface.co/docs/transformers/model_doc/detr)

## 📌 Project Overview
This project implements a full-stack **Thermal Tracking System** designed to identify and localize objects in zero-light and high-obscuration environments. By leveraging the **DEtection TRansformer (DETR)** architecture, the system achieves robust spatial reasoning on thermal heat signatures.

The model is fine-tuned on the **Teledyne FLIR ADAS Dataset**, specializing in urban traffic scenarios involving vehicles and pedestrians.

### Key Features
*   **Thermal-Specific Inference:** Specialized in processing 1-channel thermal heat maps rather than standard RGB color data.
*   **Transformer-Based Architecture:** Utilizes a ResNet-50 backbone with a Transformer encoder-decoder for end-to-end object detection.
*   **Full-Stack Deployment:** Features a high-performance **FastAPI** backend and an intuitive **Streamlit** dashboard for real-time telemetry.
*   **Privacy-Preserving:** Thermal signatures provide behavioral analytics without capturing high-resolution biometric data.

## 🏗️ Architecture


The system is divided into two distinct environments to ensure modularity and scalability:
1.  **Training Environment:** PyTorch-based training loops utilized on Kaggle for GPU-accelerated learning.
2.  **Deployment Environment:** A local web application optimized for MacBook inference with isolated virtual environments.

## 📊 Performance
*   **Optimization:** Fine-tuned for **15 Epochs** to achieve high-confidence bounding box localization.
*   **Latency:** Achieves an average inference latency of **0.45 seconds** on local hardware.
*   **Accuracy:** Capable of identifying up to **35+ concurrent objects** in complex urban frames.

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   Model weights (`thermal_detr_epoch_15.pth`)

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Hamza237516/Thermal-Tracking-Dashboard.git
    cd Thermal-Tracking-Dashboard
    ```
2.  **Set up the virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Execution
1.  **Launch Backend (FastAPI):**
    ```bash
    uvicorn api:app --reload
    ```
2.  **Launch Frontend (Streamlit):**
    ```bash
    streamlit run app.py
    ```

## 🛠️ Tech Stack
*   **Core AI:** PyTorch, HuggingFace Transformers, Timm.
*   **Web Frameworks:** FastAPI, Streamlit.
*   **Data Processing:** Pillow, NumPy, OpenCV.

---
**Developed by [Hamza Mehmood](https://github.com/Hamza237516)**A professional `README.md` acts as the front door to your project. Since you are developing this for your Master's applications, it needs to highlight both your engineering skills and your theoretical understanding of AI.

Below is a high-performance template tailored for your **Thermal-Tracking-DETR** project.

---

# 🔥 Thermal-Tracking-DETR
### *Autonomous Object Detection for Long-Wave Infrared (LWIR) Systems*

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/Model-DETR--ResNet50-orange.svg)](https://huggingface.co/docs/transformers/model_doc/detr)

## 📌 Project Overview
This project implements a full-stack **Thermal Tracking System** designed to identify and localize objects in zero-light and high-obscuration environments. By leveraging the **DEtection TRansformer (DETR)** architecture, the system achieves robust spatial reasoning on thermal heat signatures.

The model is fine-tuned on the **Teledyne FLIR ADAS Dataset**, specializing in urban traffic scenarios involving vehicles and pedestrians.

### Key Features
*   **Thermal-Specific Inference:** Specialized in processing 1-channel thermal heat maps rather than standard RGB color data.
*   **Transformer-Based Architecture:** Utilizes a ResNet-50 backbone with a Transformer encoder-decoder for end-to-end object detection.
*   **Full-Stack Deployment:** Features a high-performance **FastAPI** backend and an intuitive **Streamlit** dashboard for real-time telemetry.
*   **Privacy-Preserving:** Thermal signatures provide behavioral analytics without capturing high-resolution biometric data.

## 🏗️ Architecture


The system is divided into two distinct environments to ensure modularity and scalability:
1.  **Training Environment:** PyTorch-based training loops utilized on Kaggle for GPU-accelerated learning.
2.  **Deployment Environment:** A local web application optimized for MacBook inference with isolated virtual environments.

## 📊 Performance
*   **Optimization:** Fine-tuned for **15 Epochs** to achieve high-confidence bounding box localization.
*   **Latency:** Achieves an average inference latency of **0.45 seconds** on local hardware.
*   **Accuracy:** Capable of identifying up to **35+ concurrent objects** in complex urban frames.

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   Model weights (`thermal_detr_epoch_15.pth`)

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Hamza237516/Thermal-Tracking-Dashboard.git
    cd Thermal-Tracking-Dashboard
    ```
2.  **Set up the virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Execution
1.  **Launch Backend (FastAPI):**
    ```bash
    uvicorn api:app --reload
    ```
2.  **Launch Frontend (Streamlit):**
    ```bash
    streamlit run app.py
    ```

## 🛠️ Tech Stack
*   **Core AI:** PyTorch, HuggingFace Transformers, Timm.
*   **Web Frameworks:** FastAPI, Streamlit.
*   **Data Processing:** Pillow, NumPy, OpenCV.

---
**Developed by [Hamza Mehmood](https://github.com/Hamza237516)**
*NIT Alumnus | Aspiring MS in CSA professional `README.md` acts as the front door to your project. Since you are developing this for your Master's applications, it needs to highlight both your engineering skills and your theoretical understanding of AI.

Below is a high-performance template tailored for your **Thermal-Tracking-DETR** project.

---

# 🔥 Thermal-Tracking-DETR
### *Autonomous Object Detection for Long-Wave Infrared (LWIR) Systems*

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/Model-DETR--ResNet50-orange.svg)](https://huggingface.co/docs/transformers/model_doc/detr)

## 📌 Project Overview
This project implements a full-stack **Thermal Tracking System** designed to identify and localize objects in zero-light and high-obscuration environments. By leveraging the **DEtection TRansformer (DETR)** architecture, the system achieves robust spatial reasoning on thermal heat signatures.

The model is fine-tuned on the **Teledyne FLIR ADAS Dataset**, specializing in urban traffic scenarios involving vehicles and pedestrians.

### Key Features
*   **Thermal-Specific Inference:** Specialized in processing 1-channel thermal heat maps rather than standard RGB color data.
*   **Transformer-Based Architecture:** Utilizes a ResNet-50 backbone with a Transformer encoder-decoder for end-to-end object detection.
*   **Full-Stack Deployment:** Features a high-performance **FastAPI** backend and an intuitive **Streamlit** dashboard for real-time telemetry.
*   **Privacy-Preserving:** Thermal signatures provide behavioral analytics without capturing high-resolution biometric data.

## 🏗️ Architecture


The system is divided into two distinct environments to ensure modularity and scalability:
1.  **Training Environment:** PyTorch-based training loops utilized on Kaggle for GPU-accelerated learning.
2.  **Deployment Environment:** A local web application optimized for MacBook inference with isolated virtual environments.

## 📊 Performance
*   **Optimization:** Fine-tuned for **15 Epochs** to achieve high-confidence bounding box localization.
*   **Latency:** Achieves an average inference latency of **0.45 seconds** on local hardware.
*   **Accuracy:** Capable of identifying up to **35+ concurrent objects** in complex urban frames.

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   Model weights (`thermal_detr_epoch_15.pth`)

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Hamza237516/Thermal-Tracking-Dashboard.git
    cd Thermal-Tracking-Dashboard
    ```
2.  **Set up the virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Execution
1.  **Launch Backend (FastAPI):**
    ```bash
    uvicorn api:app --reload
    ```
2.  **Launch Frontend (Streamlit):**
    ```bash
    streamlit run app.py
    ```

## 🛠️ Tech Stack
*   **Core AI:** PyTorch, HuggingFace Transformers, Timm.
*   **Web Frameworks:** FastAPI, Streamlit.
*   **Data Processing:** Pillow, NumPy, OpenCV.

---
**Developed by [Hamza Mehmood](https://github.com/Hamza237516)**
*NIT Alumnus 
