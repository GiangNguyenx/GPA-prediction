# Student GPA Prediction Project

## Project Overview
This project aims to predict student final grades (G3) based on various demographic, social, and academic factors. The dataset includes students from Mathematics and Portuguese language courses, and the goal is to build machine learning models that can accurately predict a student's final grade.

## Dataset
The dataset contains information about students from secondary education, including:
- Demographic data (age, gender, family structure)
- Social data (relationships, free time activities)
- Academic data (study time, failures, absences)
- Educational environment data (internet access, extra educational support)

The target variable is `G3`, which represents the final grade (on a scale of 0-20).

### Data Sources
- `student-mat.csv`: Mathematics course data
- `student-por.csv`: Portuguese language course data

## Project Structure
```
GPA/
├── data/
│   ├── raw/                  # Original dataset files
│   └── processed/            # Cleaned and preprocessed data
├── models/                   # Saved model files
├── notebooks/
│   ├── eda.ipynb             # Exploratory Data Analysis
│   └── modeling.ipynb        # Model training and evaluation
├── scripts/
│   ├── etl.py                # Data extraction and preprocessing
│   └── utils.py              # Helper functions
└── README.md                 # Project documentation
```

## Setup and Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn joblib
   ```

## Data Preprocessing

The raw data undergoes several preprocessing steps:
1. Combining Mathematics and Portuguese datasets
2. Converting categorical variables to numerical values
3. Feature normalization using MinMaxScaler
4. Removing unnecessary features (school name, address)

## Models and Results

Three regression models were implemented and compared:

1. **Linear Regression**
   - MSE: 3.12
   - MAE: 1.02
   - R²: 0.798
   - Adjusted R²: 0.764

2. **Random Forest**
   - MSE: 3.03
   - MAE: 0.98
   - R²: 0.804
   - Adjusted R²: 0.771

3. **Gradient Boosting**
   - MSE: 2.64
   - MAE: 0.94
   - R²: 0.829
   - Adjusted R²: 0.801

The Gradient Boosting model performs the best with the highest R² score and lowest error metrics.

## Key Findings

- Past academic performance (G1, G2) is highly predictive of final grades
- Study time and absences are important factors affecting grades
- Socioeconomic factors show moderate correlation with academic performance
- The models can predict student grades with approximately 83% accuracy (Gradient Boosting)

## Future Improvements

- Implement more advanced models (Neural Networks, XGBoost)
- Perform feature engineering to create more informative variables
- Explore interaction effects between variables
- Use cross-validation techniques to improve model robustness
