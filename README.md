# KNN-Purchase-Prediction-App

This is a simple Streamlit app that uses a K-Nearest Neighbor (KNN) classifier to predict whether a user will purchase a product based on:

- Gender
- Age
- Estimated Salary

## Dataset

The app uses the Social Network Ads dataset from a public GitHub repository:
[Social_Network_Ads.csv](https://raw.githubusercontent.com/sakshi2k/Social_Network_Ads/refs/heads/master/Social_Network_Ads.csv)

## Features

- Gender (Male or Female)
- Age (Slider input)
- Estimated Salary (Numeric input)

## Model

- Scikit-learn's KNeighborsClassifier
- Feature scaling using StandardScaler
- Gender encoding using LabelEncoder

## How to Run

Save the script as `app.py`, then in your terminal:

```bash
streamlit run app.py
```

## Output

The app will display whether the input data corresponds to a user who is **likely to purchase** or **not purchase**, with color-coded results.

Green = ✅ Will Purchase  
Red = ❌ Will Not Purchase
