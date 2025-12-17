# FPL_ikseon (ML_FPL)

This repository contains a Flask-based prediction server (app.py) for the FPL project.
Only app.py is intended to be executed for inference.

---

## Recommended Environment

- OS: Linux (Ubuntu recommended), macOS, Windows
- Python: 3.10 (strongly recommended)

---

## 1. Clone the Repository

bash .git clone https://github.com/hanseong-star/FPL_ikseon.git .cd FPL_ikseon .

---

## 2. Create a Virtual Environment

### Option A: Conda (Recommended)

bash .conda create -n fpl python=3.10 -y .conda activate fpl .

### Option B: venv

bash .python3 -m venv .venv .source .venv/bin/activate .

---

## 3. Install Dependencies

### If requirements.txt exists

bash .pip install -r requirements.txt .

### If requirements.txt does NOT exist

bash .pip install --upgrade pip .pip install flask numpy pandas scikit-learn joblib pillow opencv-python .

If OpenCV causes GUI-related or display errors (common on servers):

bash .pip install opencv-python-headless .

---

## 4. Required Files Before Running app.py

Before running app.py, make sure all required model files and directories exist.

Typical requirements include:

- FPL_models/
 - trained model files (.pkl)
 - e.g. SVM, PCA, scaler, fusion models
- Any additional folders or configuration files referenced inside app.py

If you encounter FileNotFoundError:

- Check whether paths are relative or absolute
- Verify that filenames match exactly
- Ensure models were trained correctly
 (e.g. SVC(probability=True) if predict_proba() is used)

---

## 5. Run the Flask Server (app.py)

### Run directly with Python

bash .python app.py .

### Or using Flask CLI

bash .export FLASK_APP=app.py .export FLASK_ENV=development .flask run --host=0.0.0.0 --port=5000 .

The server will be available at:

 .http://127.0.0.1:5000/ .

---

## 6. Test the Prediction Endpoint

Example request using curl:

bash .curl -X POST http://127.0.0.1:5000/predict .

If the /predict endpoint expects an image file:

bash .curl -X POST -F "file=@/path/to/image.jpg" http://127.0.0.1:5000/predict .

Check app.py for the exact endpoint name and request format.

---

## ⚠️ IMPORTANT WARNINGS

### ❌ DO NOT run FPL.ipynb

FPL.ipynb is a training and experimentation notebook.

Running it may:

- start full training pipelines
- overwrite trained model files
- consume large amounts of CPU, RAM, or GPU
- modify dataset indexing or feature configurations

Running this notebook without full understanding can break the inference environment.

---

### ❌ DO NOT run app_mark2.py

app_mark2.py is NOT the production inference server.

It may:

- contain experimental or incomplete logic
- overwrite models or feature settings
- rely on local paths that do not exist on other machines
- produce inconsistent or invalid predictions

Only run app.py unless you are actively developing and fully understand the pipeline.

---

## Troubleshooting

### ModuleNotFoundError

bash .pip install <package-name> .

### OpenCV import or GUI errors

bash .pip install opencv-python-headless .

### predict_proba errors with SVM

- Ensure the SVM was trained with probability=True
- Or use a calibrated classifier

### Model loading failures
.
.- Check model paths and filenames used in app.py
