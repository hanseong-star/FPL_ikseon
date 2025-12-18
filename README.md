# FPL_ikseon (ML_FPL)

This repository contains a Flask-based prediction server (app.py) for the FPL project.
Only app.py is intended to be executed for inference.

---


##  Clone the Repository

bash .git clone https://github.com/hanseong-star/FPL_ikseon.git .cd FPL_ikseon .

---


##  Required Files Before Running app.py

Before running app.py, make sure all required model files and directories exist.



##  Run the Flask Server (app.py)

### Run directly with Python

bash .python app.py .

### Or using Flask CLI

bash .export FLASK_APP=app.py .export FLASK_ENV=development .flask run --host=0.0.0.0 --port=5000 .

The server will be available at:

 .http://127.0.0.1:5000/ .




### DO NOT run FPL.ipynb

FPL.ipynb is a training and experimentation notebook.

Running it may:

- start full training pipelines
- overwrite trained model files
- consume large amounts of CPU, RAM, or GPU
- modify dataset indexing or feature configurations

Running this notebook without full understanding can break the inference environment.

---

### DO NOT run app_mark2.py

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
