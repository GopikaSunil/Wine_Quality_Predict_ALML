# 🍷 Wine Quality Prediction App

## Description
A Streamlit web app to predict wine quality (Good/Bad) using a trained RandomForest model. Supports **manual input** and **CSV upload** for batch predictions.

### 1. Clone repo
git clone <repo-url>
cd wine-quality-app

---

### 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

---

### 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

---

### 4. Train the model (optional if wine_model.pkl not present)
python model.py

---

### 5. Run Streamlit app in background (nohup) and save logs
nohup streamlit run app.py --server.address 0.0.0.0 --server.port 8080 > log.txt 2>&1 &

---
### Run this in browser
http://<EC2-PUBLIC-IP>:8080

---

### Note
Ensure wine_model.pkl and winequality.csv are in the same folder as app.py.
For cloud deployment, make sure EC2 Security Group allows the port used.

