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
│   ├── building_segmentation.ipynb     # Building segmentation model training & inference
│   ├── change_detection.ipynb       # Change detection model using temporal imagery
│   ├── waterbody_segmentation.ipynb    # Waterbody segmentation model training & inference
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

## Models Used

All models used in this project are based on the **U-Net architecture**, a widely adopted convolutional network for semantic segmentation. Each model was tailored to its specific task, with variations in input shape, depth, and number of filters to balance performance and efficiency.

### 🔹 1. Building Detection Model

* **Architecture**: Optimized U-Net
* **Input Shape**: `(640, 640, 3)`
* **Task**: Binary segmentation to identify building footprints in aerial imagery.
* **Characteristics**:
  * Lightweight encoder-decoder design with gradual feature scaling (16 → 256 filters).
  * Batch Normalization and Dropout regularization for better generalization.
  * Output: Binary mask highlighting building regions.
* **Loss**: Binary Cross-Entropy / Dice Loss (depending on training setup)

> Designed for faster inference on high-resolution satellite/drone `.tif` images while maintaining decent precision.

### 🔹 2. Change Detection Model

* **Architecture**: Deep U-Net with higher capacity
* **Input Shape**: `(512, 512, 6)` — stacked pair of pre-change and post-change RGB images
* **Task**: Segment areas that have undergone changes (e.g., new constructions)
* **Characteristics**:
  * Deeper network with a larger bottleneck (up to 512 filters)
  * Takes concatenated temporal image pairs as input
  * Useful for flagging regions to be further analyzed for new or unauthorized structures

> This model helps automatically narrow down areas of interest by comparing different timestamps of the same location.

### 🔹 3. Waterbody Detection Model

* **Architecture**: Classic U-Net with wide layers
* **Input Shape**: `(640, 640, 3)`
* **Task**: Semantic segmentation of waterbodies such as lakes, rivers, or ponds.
* **Characteristics**:
  * Higher parameter count with up to 1024 filters at the bottleneck
  * Accurate delineation of water edges and proximity regions
  * Essential for proximity analysis during environmental rule checks

> This model ensures accurate assessment of construction legality near waterbodies.

Each model was trained and validated independently with task-specific data preprocessing and augmentation. You can find the corresponding training code and sample predictions in the `notebooks/` folder.

---

## Acknowledgements

We'd like to acknowledge the following resources and frameworks:

* TensorFlow & Keras for model development
* OpenCV, Rasterio and GeoPy for image and geo-data processing
* Publicly available datasets for aerial imagery
* Team Nirman Netra for end-to-end development of the system

---
