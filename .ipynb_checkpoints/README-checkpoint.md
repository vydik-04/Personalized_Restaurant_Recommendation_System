# Personalized Restaurant Recommendation System

This repository contains a machine learning pipeline designed to predict which restaurants (vendors) customers are most likely to order from. Recommendations are based on a combination of **customer demographics**, **vendor attributes**, and **spatial proximity (locations)**.

## Project Overview

The objective of this project is to build a robust recommendation engine for food delivery. Given a customer's location, profile, and order history, the engine ranks vendors they are most likely to interact with. 

### Key Innovations in this Pipeline
1. **Meaningful Target Formulation:** Order interactions are aggregated at the `(Customer, Vendor)` pair level. Instead of predicting individual noisy orders, the model predicts the likelihood of an interaction between a pair.
2. **Balanced Negative Sampling:** The model uses a 1:1 ratio of positive samples (restaurants the user actually ordered from) and negative samples (restaurants the user didn't order from), forcing it to learn actual preferences.
3. **Leakage-Free Feature Engineering:** The pipeline strictly avoids look-ahead bias by completely stripping out delivery-specific details (like delivery time, driver ratings) that are only available *after* an order is placed.
4. **Cold-Start Resilience:** The test dataset features "zero overlap" customers (a cold-start scenario). The model relies heavily on universal predictors—vendor popularity, distance between customer and vendor, and general demographic tendencies—to deliver high-quality recommendations for completely new users.

## Directory Structure

```text
📁 Personalized Restaurant Recommendation System/
├── 📁 Assignment/
├── 📁 Datasets/
│   ├── 📁 Train/ 
│   │   ├── orders.csv
│   │   ├── train_customers.csv
│   │   ├── train_locations.csv
│   │   └── vendors.csv
│   ├── 📁 Test/
│   │   ├── test_customers.csv
│   │   └── test_locations.csv
│   ├── SampleSubmission.csv
│   └── VariableDefinitions.pdf
├── 📓 personalized_restaurant_recommender.ipynb  # Primary ML pipeline
├── 📄 submission.csv                             # Final generated predictions
└── 📄 README.md                                  # You are here
```

## How It Works

1. **Data Preprocessing:** Parses missing variables and masks missing fields with `Unknown` or median values automatically. 
2. **Distance Calculation:** A vital feature is the computed Euclidean distance based on the masked `latitude` and `longitude` fields between customers and vendors.
3. **Feature Selection:** Trims the dataset down to 25 highly meaningful, prediction-time-safe features.
4. **Encoding & Training:** Uses `LabelEncoder` to encode categorical inputs, retaining memory mapping for testing. An `XGBClassifier` models the data to generate recommendation probabilities.
5. **Inference & Submission:** Reconstructs the exact required `[Customer ID] X [Location Number] X [Vendor ID]` strings mapped to their probabilities to yield precisely 1,672,000 predictions.

## Local Setup & Usage

### Prerequisites
- Python 3.8+
- [Jupyter Notebook](https://jupyter.org/install) 

### Dependencies
Install the required libraries:
```bash
pip install pandas numpy xgboost scikit-learn matplotlib seaborn
```

### Running the Pipeline
Open the primary notebook in your Jupyter environment:
```bash
jupyter notebook personalized_restaurant_recommender.ipynb
```
Select **Run All Cells**. The notebook is optimized and scales effectively. Once completed, the notebook will automatically generate `submission.csv` at the root of the project folder.

## Evaluation & Performance
On the training validation sets, 3-Fold Cross-Validation achieved:
* **Mean ROC-AUC**: `~0.8182`

**Top 5 Predictive Features Derived:**
1. `vendor_popularity` (Global restaurant success)
2. `distance` (Proximity of the restaurant to the user location)
3. `vendor_latitude` / `vendor_longitude` (Restaurant spatial positioning)
4. `prepration_time` (Speed of delivery readiness)
5. `delivery_charge` (Cost to the user)

## License
Provided for internal assessment and assignment execution purposes.
