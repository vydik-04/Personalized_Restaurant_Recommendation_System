# 🍽️ Personalized Restaurant Recommendation System
    This project implements a machine learning-based recommendation engine that predicts which restaurants (vendors) a customer is most likely to order from. The system leverages customer behavior, vendor characteristics, and spatial features to generate personalized recommendations.
---

## 📌 Project Objective
    The goal is to build a predictive model that ranks vendors for each `(Customer, Location)` pair based on the likelihood of an order.

    Given:
        - Customer information
        - Location data
        - Vendor attributes
        - Historical order data

    The model outputs:

        CID X LOC_NUM X VENDOR → probability of order


---

## 🧠 Key Concepts Used

    - Data Preprocessing & Cleaning  
    - Feature Engineering  
    - Recommendation System Logic  
    - Classification using Machine Learning  
    - Handling Real-world Data Issues (missing values, encoding, mismatches)

---

## ⚙️ Pipeline Overview

### 1. Data Loading & Merging
    Multiple datasets are merged:
        - `orders.csv`
        - `train_customers.csv`
        - `train_locations.csv`
        - `vendors.csv`

---

### 2. Data Cleaning
    - Dropped highly sparse/unreliable columns  
    - Handled missing values using:
        - Median (numerical)
        - "Unknown" (categorical)

---

### 3. Feature Engineering (🔥 Core Strength)

    Key features created:

        - 📍 **Distance**
        - Euclidean distance between customer and vendor

        - ⏰ **Time Features**
        - Order hour  
        - Day  
        - Weekday  

        - 👤 **Customer Behavior**
        - Total orders per customer  

        - 🍽️ **Vendor Popularity**
        - Total orders per vendor  

        - ❤️ **Customer-Vendor Interaction**
        - Number of times a customer ordered from a vendor  

---

### 4. Target Engineering

    - Positive samples: actual orders → `target = 1`  
    - Negative samples: randomly generated non-ordered vendors → `target = 0`  

    This allows the model to learn real preferences.

---

### 5. Encoding

    - Categorical features encoded using `LabelEncoder`
    - Ensured consistency between train and test data
    - Handled unseen categories safely

---

### 6. Model Training

    Model used:
    ```python
    XGBClassifier

    Why?

    Handles structured data well
    Captures non-linear relationships
    Works effectively with engineered features
    7. Prediction
    Generated probabilities using:
    predict_proba()
    Ensured:
    No object dtype issues
    Train-test column alignment
    8. Submission Format

### Final output:

    CID X LOC_NUM X VENDOR    target
    Z59FTQD X 0 X 243         0.82

    📊 Key Features Driving Predictions:
        Vendor popularity
        Distance between customer & vendor
        Customer order frequency
        Customer-vendor relationship strength
        Time-based ordering patterns

    ⚠️ Challenges Faced & Solutions:
        Challenge	Solution
        Model predicting constant values	Added negative sampling
        Train-test mismatch	Column alignment
        Encoding errors	Consistent encoders
        XGBoost dtype errors	Forced numeric conversion
        Broken IDs in submission	Preserved original test data
        📈 Model Performance (Practical View)
        Captures meaningful behavioral patterns
        Performs significantly better than random guessing
        Not fully optimized (no hyperparameter tuning / ranking metrics)

    🚀 How to Run:
        Install dependencies:
            pip install pandas numpy xgboost scikit-learn matplotlib seaborn
        Run:
            jupyter notebook personalized_restaurant_recommender.ipynb

    🧠 Learning Outcomes:
        This project demonstrates:
            Building an end-to-end ML pipeline
            Working with real-world messy data
            Designing recommendation logic
            Debugging complex ML issues
            Understanding model limitations

    🎯 Future Improvements:
        Use ranking models (LightGBM Ranker)
        Add cuisine-based features
        Evaluate using Precision@K / Recall@K
        Hyperparameter tuning
        Deploy using Streamlit

    📌 Conclusion:
        This project showcases a complete machine learning workflow—from raw data to a working recommendation system. While not fully production-optimized, it provides a strong foundation in real-world data science practices.