# Nirman_Netra_Models

**Nirman Netra** is an AI-powered monitoring platform for detecting unauthorized constructions using aerial imagery. This repository focuses on the **deep learning model development** aspect of the project. It includes:

* A **building detection model** that identifies structures from high-resolution drone or satellite imagery.
* A **waterbody detection model** to detect lakes, rivers, or ponds near constructions.
* A **change detection model** to flag newly constructed areas over time by comparing temporal aerial images.

All models were trained on `.tif` format images containing geo-metadata, enabling further geospatial analysis such as:

* Estimating **building dimensions**
* Measuring **proximity to waterbodies**
* Resolving **geographic addresses**
* Flagging **unauthorized buildings** by integrating with regulation data and assumed government databases.

This repository contains only the model development notebooks and sample outputs. The full application (frontend + backend) is maintained in a separate repository.

---

## 🗂️ Project Structure

```bash
Nirman_Netra_Models/
├── notebooks/
│   ├── building_detection.ipynb     # Building detection model training & inference
│   ├── change_detection.ipynb       # Change detection model using temporal imagery
│   ├── waterbody_detection.ipynb    # Waterbody detection model
├── sample_outputs/
│   ├── building_predictions.png
│   ├── change_map_example.png
│   └── waterbody_overlay.png
├── epoch_details/
│   ├──
│   ├──
│   └──
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies (optional)
```

---

## How to Run

> These notebooks are designed to be run on a Python environment (recommended: Python 3.10), preferably with GPU support for faster training/inference.

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Janesh-e/Nirman_Netra_Models.git
   cd Nirman_Netra_Models
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Open and run notebooks**:

   * `building_detection.ipynb`: Trains and tests a model to detect buildings from aerial imagery.
   * `change_detection.ipynb`: Compares two georeferenced images to identify changed areas.
   * `waterbody_detection.ipynb`: Detects waterbodies in aerial views to assess environmental impact.

> All notebooks include inline documentation and examples using sample `.tif` files (with embedded geo-metadata).

---

## Acknowledgements

We'd like to acknowledge the following resources and frameworks:

* TensorFlow & Keras for model development
* OpenCV, Rasterio and GeoPy for image and geo-data processing
* Publicly available datasets for aerial imagery
* Team Nirman Netra for end-to-end development of the system

---
