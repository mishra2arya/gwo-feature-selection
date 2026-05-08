# 🐺 GWO Feature Selection on Madelon Dataset

A machine learning project that uses the **Grey Wolf Optimizer (GWO)** metaheuristic algorithm to perform feature selection on the Madelon dataset, improving classification accuracy while reducing the number of input features.

---

## 📌 Project Overview

| Step | Description |
|------|-------------|
| Dataset | Madelon (OpenML) — 2000 samples, 500 features |
| Baseline | SVM with RBF kernel on all features |
| Feature Selection | GWO (binary) + SelectKBest (ANOVA & Mutual Info) |
| Final Model | Random Forest Classifier (500 trees) |

---

## 🧠 How It Works

1. **Load & preprocess** the Madelon dataset (standardize with `StandardScaler`)
2. **Baseline SVM** is trained on all features to establish a reference accuracy
3. **SelectKBest (ANOVA)** pre-filters to top 50 features
4. **Grey Wolf Optimizer (GWO)** evolves a binary population to find the best feature subset, using a penalized fitness function: `(1 - accuracy) + 0.02 × feature_ratio`
5. **Final Random Forest** is trained on top 150 features selected via Mutual Information

---

## 📊 Results

- ✅ GWO converges over 30 iterations, minimizing the fitness score
- ✅ Feature count reduced significantly from 500
- ✅ Final Random Forest accuracy compared against baseline SVM

---

## 🗂️ Project Structure

```
gwo-feature-selection/
├── gwo_feature_selection.ipynb   # Main Jupyter notebook
├── requirements.txt              # Python dependencies
├── .gitignore                    # Files to ignore
└── README.md                     # This file
```

---

## ⚙️ Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/gwo-feature-selection.git
cd gwo-feature-selection
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter and run the notebook
```bash
jupyter notebook gwo_feature_selection.ipynb
```

> ⚠️ The first run downloads the Madelon dataset (~10MB) from OpenML into `openml_cache/`. This is automatic.

---

## 📦 Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

See `requirements.txt` for pinned versions.

---

## 📖 References

- [Grey Wolf Optimizer — Mirjalili et al. (2014)](https://doi.org/10.1016/j.advengsoft.2013.12.007)
- [Madelon Dataset — OpenML](https://www.openml.org/d/1485)
- [scikit-learn Documentation](https://scikit-learn.org)

---

## 📝 License

This project is for academic/educational use.
