# Thyroid Disorder Detection System 🧠🩺

A final-year project that uses machine learning to predict thyroid disorders (like hypothyroidism) based on patient data.

## 📌 Project Highlights

- Predicts thyroid status using clinical data.
- Trained ML model (`thyroid_model.pkl`) based on the `hypothyroid.csv` dataset.
- SQLite database for storing login and prediction history.
- Simple login system via `login.py`.
- Jupyter Notebook (`newone.ipynb`) shows data cleaning, model training, and evaluation.

## 🛠 Requirements

- Python 3.x
- Libraries: `pandas`, `scikit-learn`, `numpy`, `joblib`, `sqlite3`
- [Optional] `jupyter` for notebook

Install dependencies:
```bash
pip install -r requirements.txt
```
# 🔬 Thyroid Disorder Detection – Final Year Project

## ▶️ How to Run the Project

1. **Clone this repository**
   ```bash
   git clone https://github.com/Addy20021/finalyearproject.git
   cd finalyearproject
   ```

2. **Set up the SQLite database**
   ```bash
   python setup_db.py
   ```

3. **Run the login and prediction interface**
   ```bash
   python login.py
   ```

4. **(Optional)** Retrain or tweak the model  
   Open and run the Jupyter notebook:
   ```bash
   jupyter notebook newone.ipynb
   ```

---

## 📁 File Descriptions

| File                     | Purpose                                              |
|--------------------------|------------------------------------------------------|
| `hypothyroid.csv`        | Clinical dataset used for training                   |
| `thyroid_model.pkl`      | Pre-trained ML model (joblib format)                 |
| `login.py`               | User login and input interface                       |
| `setup_db.py`            | Initializes `thyroid.db` with tables                 |
| `thyroid.db`             | SQLite database storing users & logs                 |
| `newone.ipynb`           | Model training notebook                              |
| `gonna_use at the end.py`| Final script for running predictions                 |
| `str.py`                 | Utility functions for input handling                 |
| `logo.jpg`, `background.jpg` | Visual assets for UI or docs                 |
| `contacts.csv`           | Sample contact file (optional use)                  |

---

## 📊 Model Summary

- **Dataset**: `hypothyroid.csv`  
- **Algorithm**: Random Forest / Decision Tree *(as implemented in the notebook)*  
- **Target**: Classification of thyroid disorder type  
- **Accuracy**: Printed after model evaluation in `newone.ipynb`  

---

## 🔐 Login Credentials

- Credentials can be preset or modified using `setup_db.py`.
- Default credentials (if any) are displayed when the script is run.

---

## ✅ Future Improvements

- Web-based UI using **Flask** or **Streamlit**
- Password hashing and secure login system
- Enhanced data visualizations
- PDF/CSV report export for patients/doctors

---

## 👥 Authors

| Name               | Contact |
|--------------------|--------|
| **Adarsh Shetty**  |  [GitHub](https://github.com/Addy20021) |
| **Chirasmita Salian** |  [Email](mailto:chirasmita06@gmail.com) |

