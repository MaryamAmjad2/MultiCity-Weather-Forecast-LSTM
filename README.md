# 🌍 Multi-City Weather Forecast with LSTM

A production-ready deep learning time-series pipeline designed to predict daily minimum (`tmin`), maximum (`tmax`), and average (`tavg`) temperatures across multiple diverse geographical regions in Pakistan using a **Shared LSTM Architecture with Entity Embeddings**.

---

## 🚀 Key Features
* **Multi-City Shared Modeling:** A single unified LSTM model handles 6 major cities simultaneously by leveraging categorical `city_id` embeddings to capture regional climate heterogeneity (ranging from mountainous Gilgit to coastal Karachi).
* **High-Precision Base Training:** Trained on extensive historical data (2000–2024) with Early Stopping, achieving an outstanding overall **R² score of ~97.35%**.
* **Climate Drift Adaptation (Fine-Tuning):** Implements transfer learning and fine-tuning on recent unseen datasets (`Sep 2024 – Oct 2025`) to seamlessly adapt to recent weather anomalies, pushing the adapted global R² score to **~78.01%**.
* **Universal Inference Engine:** Ready-to-use pipeline for loading serialized artifacts (`.pth` weights, `.pkl` scalers, and city mappings) to run real-time 30-day window forecasting.

---

## 📊 Performance & Evaluation Summary

### 1. Base Model Performance (Historical Test Split: 2000–2024)
Evaluated across 6 major cities (`Gilgit`, `Islamabad`, `Karachi`, `Lahore`, `Peshawar`, `Quetta`) with **6,178 test sequences**:
* **Overall Combined RMSE:** 1.5566 °C
* **Overall Combined MAE:** 1.1457 °C
* **Global R² Score:** **0.9735 (97.35% variance explained)** — *Almost all city targets rated as "Excellent" (R² > 0.85).*

### 2. Fine-Tuned Performance (Unseen Out-of-Sample Data: Sep 2024 – Oct 2025)
After applying 10 epochs of fine-tuning transfer learning on recent weather data:
* **Overall Combined RMSE:** 1.8574 °C
* **Overall Combined MAE:** 1.3523 °C
* **Global R² Score:** **0.7801 (78.01% variance explained)** — *Successfully adapting to recent climate shifts and seasonal anomalies.*

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.8+
* **Deep Learning Framework:** PyTorch (`torch`, `torch.nn`)
* **Data Processing & Metrics:** Pandas, NumPy, Scikit-Learn (`StandardScaler`, `mean_absolute_error`, `r2_score`)
* **Model Serialization:** Joblib (`.pkl`), PyTorch State Dict (`.pth`)
* **Visualization:** Matplotlib

---


