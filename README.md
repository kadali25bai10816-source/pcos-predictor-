# 🩺 PCOS / PCOD Risk Prediction System (Machine Learning Project)

## 📌 Project Overview

This project is a **machine learning-based PCOS/PCOD risk prediction system** developed using Python. The system takes user health data and symptoms as input and predicts the probability of Polycystic Ovary Syndrome (PCOS) using a trained Random Forest model.

The project also includes an **interactive symptom survey**, **automatic BMI calculation**, and **visual risk analysis using graphs**. It is designed as a beginner-friendly medical AI project suitable for students in B.Tech (CSE/AI-ML) who want to build real-world healthcare-based applications.

---

## 🎯 Objectives

* Predict the risk of PCOS using machine learning
* Allow users to enter symptoms interactively
* Automatically calculate BMI from height and weight
* Display prediction results clearly (LOW / MODERATE / HIGH risk)
* Visualize results using charts
* Save the trained model for future use

---

## 🧠 Machine Learning Model Used

The system uses:

* **Random Forest Classifier** for prediction
* **StandardScaler** for feature scaling
* **Train-Test Split** for model evaluation

The model is trained on a generated medical-style dataset containing symptoms such as:

* Irregular periods
* Acne
* Hair loss
* Weight gain
* Infertility
* Insulin resistance
* High androgen levels
* Ovarian cysts
* Family history
* And many more

---

## ⚙️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib

---

## 📂 Project Structure

```
PCOS-Prediction-System/
│
├── pcos_predictor.py          # Main project file
├── pcos_predictor_model.pkl   # Saved trained model
└── README.md                  # Project documentation
```

---

## 🚀 How to Run the Project

### Step 1: Install required libraries

Run this command in terminal:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

---

### Step 2: Run the Python file

```bash
python pcos_predictor.py
```

---

### Step 3: Enter your details

The program will ask:

* Age
* Weight
* Height
* Symptoms (Yes = 1 / No = 0)

---

### Step 4: Get Prediction

The system will show:

* PCOS Prediction (YES / NO)
* Risk Percentage
* Risk Level (LOW / MODERATE / HIGH)
* Top important symptoms
* Graphical result (pie chart + symptom chart)

---

## 📊 Features of the Project

* Interactive command-line interface
* BMI calculation
* Machine learning prediction
* Risk probability output
* Symptom importance detection
* Graphical visualization
* Model saving using Joblib

---

## 💡 Future Improvements

This project can be improved by:

* Adding a GUI using Tkinter or Streamlit
* Using a real medical dataset
* Deploying the project on a website
* Adding more health parameters
* Creating a mobile app version

---

## 🎓 Suitable For

This project is perfect for:

* B.Tech CSE (AI & ML) students
* Beginners in Machine Learning
* Healthcare-based AI projects
* Mini Projects / First-Year Projects
* Portfolio and GitHub showcase

---

## ⚠️ Disclaimer

This project is created only for **educational purposes**. It does not replace medical diagnosis. Always consult a qualified doctor for proper medical advice.
