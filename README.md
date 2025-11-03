# 🩸 Diabetes Prediction

This project is an end-to-end diabetes prediction app.

* A trained machine learning model that takes in patient health metrics and predicts **"Diabetes"** or **"No Diabetes"**.
* A lightweight web interface where you can enter values and get an instant prediction.
* A Python backend API that loads the trained model and runs inference in real time.
* All model artifacts are included (`my_model.keras`, `scaler.joblib`) so anyone can reproduce your results locally.
* A **training notebook** (`create_model.ipynb`) that shows exactly how the artifacts were produced (reproducible).

The goal: make diabetes screening logic accessible and testable in seconds, not stuck in a notebook.

---

## 🔴 Demo

https://github.com/user-attachments/assets/2ae88c79-64b7-4b99-b13a-4cfe0743c426

**What the demo shows:**

1. You enter patient information (glucose level, BMI, etc.).
2. You click “Predict.”
3. The system runs the trained model.
4. It returns either **"Diabetes"** or **"No Diabetes"**, and can optionally include a confidence score.

That’s a full ML → app pipeline.

---

## 🧠 What this project does

Diabetes is very common and often undiagnosed. The idea here is not to diagnose (this is not a medical device), but to **screen** based on known risk indicators.

This project:

1. Accepts structured health inputs.
2. Applies the same preprocessing used during training (scaling, normalization).
3. Runs a trained model on those inputs.
4. Returns a binary prediction:

   * `"Diabetes"`
   * `"No Diabetes"`

Because the backend exposes a simple API, this can be embedded in other tools (triage dashboard, intake form, etc.).

---

## 🏗 System Overview

This is how the pieces of your repo work together:

```text
[ interface.html ]           <-- browser UI
      |
      |  sends the form data (patient metrics) to the backend
      v
[ server/server.py ]         <-- inference API
      |
      |  loads trained model + scaler from disk
      |  transforms input, runs prediction
      v
[ server/artifacts/ ]        <-- trained ML assets
    - my_model.keras         (saved Keras model)
    - scaler.joblib          (fitted scaler from training)
```

Also in `server/`:

* `requirements.txt` lets anyone recreate the same Python env.
* `prediction.csv` can store sample data, logs, or testing runs.

The flow is:

**User enters numbers → frontend POSTs → backend scales & predicts → frontend shows `"Diabetes"` or `"No Diabetes"`.**

---

## ✨ Features

### 🩺 Binary diabetes screening

The model’s output is clean and human-readable:

* **"Diabetes"** → the model believes this input pattern matches diabetic cases.
* **"No Diabetes"** → the model believes this looks non-diabetic.

No vague "high risk / low risk" wording. You get a straight classification.

### 🌐 API-backed interface

* `interface.html` is not just a static page — it talks to the backend.
* When you click “Predict,” the page sends the input values to the server.
* The server returns something like:

  ```json
  {
    "prediction": "Diabetes",
    "confidence": 0.87
  }
  ```

  or:

  ```json
  {
    "prediction": "No Diabetes",
    "confidence": 0.94
  }
  ```

You can display that result directly in the UI.

### 📦 Checked-in artifacts

Inside `server/artifacts/`:

* `my_model.keras`

  * The trained neural network / ML model.
* `scaler.joblib`

  * The numeric feature scaler that was fit on the training data.

During inference, the backend:

1. Takes raw inputs from the user.
2. Uses `scaler.joblib` to transform them to the same numeric scale the model was trained on.
3. Feeds that into `my_model.keras`.
4. Interprets the output as "Diabetes" or "No Diabetes".

That means predictions are consistent with training. You're not guessing.

### 📝 Reusable backend service

`server/server.py` is a small model-serving API:

* It loads everything once at startup.
* It exposes an endpoint (like `/predict`) that other code can call.
* That makes this thing usable from the HTML front-end, Postman, curl, or any other app.

This is what “productionizing” a model looks like at a small scale.

---

## 📂 Repository Structure

Here’s your current layout:

```text
diabetes-prediction/
├─ README.md                 # <-- this file
├─ create_model.ipynb        # Mode training and testing before deployment             
├─ interface.html            # Web UI to enter patient metrics + view prediction
└─ server/
   ├─ server.py              # Backend inference API (Flask/FastAPI style)
   ├─ requirements.txt       # Python dependencies
   ├─ prediction.csv         # Example/record of predictions (optional)
   └─ artifacts/
      ├─ my_model.keras      # Trained diabetes / no-diabetes classifier
      └─ scaler.joblib       # Feature scaler used during training
```

### Why this layout is strong

* `interface.html` = front-end (human-facing)
* `server/` = back-end service (machine-facing)
* `artifacts/` = trained ML assets (model + scaler)

That separation makes it super obvious how to reuse this in another product.

---

## 🔌 Backend API (server/server.py)

The backend is responsible for:

1. Loading `my_model.keras` and `scaler.joblib`.
2. Accepting raw numeric inputs from the user.
3. Building the feature vector in the correct order (glucose, BMI, age, etc.).
4. Scaling that vector.
5. Running `model.predict(...)`.
6. Converting that output into:

   * `"Diabetes"` or
   * `"No Diabetes"`
     plus an optional confidence score.

### Example request body (conceptual):

```json
{
  "pregnancies": 2,
  "glucose": 145,
  "blood_pressure": 82,
  "skin_thickness": 30,
  "insulin": 90,
  "bmi": 33.1,
  "dpf": 0.45,
  "age": 41
}
```

### Example response:

```json
{
  "prediction": "Diabetes",
  "confidence": 0.87
}
```

Or:

```json
{
  "prediction": "No Diabetes",
  "confidence": 0.94
}
```

Your exact field names might differ. Update them here if needed so it matches `server.py`.

---

## 🖥 Frontend Flow (interface.html)

How a user experiences it:

1. They open `interface.html`.
2. They see a form with fields like glucose, BMI, age, etc.
3. They press “Predict.”
4. The page sends those numbers to the backend `/predict` endpoint.
5. The backend responds with `"Diabetes"` or `"No Diabetes"` (and a confidence).
6. The page displays that result clearly, for example:

   * red badge “Diabetes”
   * or green badge “No Diabetes”

This turns “just a model” into an actual tool.

---

## 🧪 Model Trainer (create_model.ipynb)

The notebook create_model.ipynb fully reproduces the model artifacts used by the backend.

### Data split

- 60% / 20% / 20% = train / validation / test (stratified).

- All preprocessing steps fit only on the training split; val/test are transformed using the fitted objects.

### Preprocessing & Feature Engineering

- **Median replacement for missing/invalid numeric entries**

  - Robust against skewed distributions.

- **Feature selection (correlation-based)**

  - Compute Pearson correlation with the label (and among features) to drop low-signal or highly collinear inputs.
    
  <p align="center"> <img width="932" height="849" alt="Image" src="https://github.com/user-attachments/assets/824212f9-3eea-467d-b036-753fae8d0130" /> </p>

- **StandardScaler normalization**

- **Class balance with SMOTE**

  - Handle class imbalance by generating synthetic minority samples after splitting (fit SMOTE on train only).
  - Adding more diabetic samples to balance them.

  <p align="center"> <img width="571" height="526" alt="Image" src="https://github.com/user-attachments/assets/618915b0-079e-4f3c-a9c1-2a682e28b32b" />
  <img width="571" height="526" alt="Image" src="https://github.com/user-attachments/assets/3bf7ced5-999a-4638-a19e-167b9ccae254" /></p>
### Model architectures (three variants)

Train and compare three compact neural networks. Example definitions (high-level):

- **First Neural Network (Baseline, 32 Neurons, 1 Hidden Layer)**

  - Dense(32, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)

- **Second Neural Network (64 Neurons, 1 Hidden Layer)**

  - Dense(64, ReLU) → Dropout(0.2) → Dense(1, Sigmoid)

- **Third Neural Network (Deeper, 64 and 32 Neurons, 2 Hidden Layers)**

  - Dense(64, ReLU) → Dropout(0.5)

  - Dense(32, ReLU) → Dropout(0.3)

  - Dense(1, Sigmoid)

**Training setup (consistent across models)**

  - Loss: BinaryCrossentropy

  - Optimizer: Adam

  - Metrics: Accuracy, Precision, Recall, F1

  - Callbacks: EarlyStopping (val loss)

### Evaluation & selection

  - Evaluate all three on **validation**, pick the best, then report on **test**.

  - Display the three confusion matrices, and based on that the second model variant had the least false negatives which means it can predict more diabetic cases accurately.


<p align="center"><b>Confusion Matrices</b></p>

<table align="center">
  <tr>
    <th>Model Variant 1</th>
    <th>Model Variant 2</th>
    <th>Model Variant 3</th>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/98796f6b-cc59-4b14-a6ac-0f2ff3789e0c" width="300" alt="Confusion Matrix - Model Variant 1"><br/>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/5bb5be19-f68e-4031-93f1-9a0a3a8b97a1" width="300" alt="Confusion Matrix - Model Variant 2"><br/>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/5c320ac8-fe82-452a-bd66-321ea071d565" width="300" alt="Confusion Matrix - Model Variant 3"><br/>
    </td>
  </tr>
</table>

### Model Comparison (Neural Network Variants)

| Variant | Train Acc | Val Acc | Test Acc | Precision (Diabetic) | Recall (Diabetic) | F1 (Diabetic) |
|:------:|:---------:|:-------:|:--------:|:---------------------:|:-----------------:|:-------------:|
| **V1** | 0.7375 | 0.7403 | 0.7792 | 0.66 | 0.76 | 0.71 |
| **V2** | **0.7525** | **0.7532** | **0.8117** | **0.70** | **0.80** | **0.75** |
| **V3** | 0.7475 | 0.7403 | 0.7857 | 0.68 | 0.74 | 0.71 |

**Interpretation**
- **V2 is the best overall**: highest validation/test accuracy and strongest F1 for the *Diabetic* class, indicating the best balance of precision and recall.
- **V1 (baseline)** shows **underfitting**: lower test performance and weaker precision/recall balance.
- **V3** shows **slight overfitting**: train accuracy is fine but validation/test metrics don’t improve and F1 drops vs. V2.

*Why these metrics?*  
- **Precision (Diabetic)** = when the model predicts *diabetic*, how often it’s correct.  
- **Recall (Diabetic)** = how many true diabetic cases the model catches.  
- **F1 (Diabetic)** = harmonic mean of precision & recall a balanced measure for imbalanced data.

---

## 🚀 Run it locally

### 1. Set up Python env for the backend

```bash
cd server
python -m venv venv

# macOS / Linux:
source venv/bin/activate

# Windows:
# venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Start the backend API

```bash
python server.py
```

That should start a local server (for example `http://localhost:5000`, depending on your code).

### 3. Open the interface

* Open `interface.html` in your browser.
* Make sure the JavaScript inside it is pointing to your backend URL, e.g. `http://localhost:5000/predict`.
* Enter some values and test a prediction.

Once that round-trip works, you're done: you’ve got a working local mini “diabetes screener.”

---

## 🛣 Roadmap / future improvements

* [ ] Add input validation (catch missing or impossible values).
* [ ] Show which features drove the prediction (explainability / feature importance).
* [ ] Batch upload: allow CSV upload to predict multiple patients at once.
* [ ] Deploy to a cloud host or container so nurses / intake staff can use it without running Python.
* [ ] Store prediction history securely (audit trail).

---

## ⚠️ Important notes

* This is **not** a medical device, and it is **not** a diagnosis.
  It’s a machine learning model that predicts patterns similar to diabetes cases.
* Real clinical decisions should always involve licensed medical professionals, labs, and proper testing.
* Do not store or share real patient identifiers in `prediction.csv` or anywhere else.

---

## TL;DR for reviewers / hiring managers

**What this repo shows:**

* Trained a diabetes / no-diabetes classifier.
* Exported the trained model (`my_model.keras`) and its scaler (`scaler.joblib`).
* Built a backend service that can take real inputs and run inference.
* Built a frontend to make that usable to a normal person.
* Recorded a working demo.
* Added `create_model.ipynb` to show exact training steps end-to-end.

That’s the full ML product loop:
**train → package → serve → UI → demo.**
