# 🩸 Diabetes Prediction

This project is an end-to-end diabetes prediction app.

* A trained machine learning model that takes in patient health metrics and predicts **"Diabetes"** or **"No Diabetes"**.
* A lightweight web interface where you can enter values and get an instant prediction.
* A Python backend API that loads the trained model and runs inference in real time.
* All model artifacts are included (`my_model.keras`, `scaler.joblib`) so anyone can reproduce your results locally.

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
├─  create_model.ipynb       # Mode training and testing before deployment             
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

That’s the full ML product loop:
**train → package → serve → UI → demo.**
