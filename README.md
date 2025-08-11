# CS687 Capstone – Credit Card Fraud Detection with MLOPs Integration

Initial setup complete. Data preprocessing and cleaning started.

Credit Card Fraud Detection – Data Exploration & Preprocessing

Overview
This script performs exploratory data analysis (EDA) and preprocessing on a credit card transaction dataset. The goal is to understand the dataset, identify imbalances, visualize patterns, and prepare it for machine learning model training.

Steps Performed
1. Import Required Libraries
pandas, numpy – Data handling

matplotlib, seaborn – Visualization

sklearn.preprocessing – Scaling features

2. Load Dataset
python
Copy
Edit
df = pd.read_csv("creditcard.csv")
Reads the dataset into a pandas DataFrame.

3. Overview of Dataset
Shape of dataset – (rows, columns)

Info summary – column names, data types, null values.

4. Display First Few Rows
Shows first 5 records for a quick view.

5. Missing Values Check
Counts missing values in each column.

6. Summary Statistics
Uses .describe() to get count, mean, std, min, max, and percentiles for numeric columns.

7. Class Distribution Analysis
Checks fraud vs non-fraud transactions.

Visualizes using a count plot to highlight imbalance.

8. Correlation Matrix
Visualizes correlations between:

Time

Amount

Class

Uses a heatmap for clarity.

9. Duplicate Records Check
Identifies and optionally removes duplicates.

10. Feature Distribution
Plots distribution of the Amount feature.

11. Feature Scaling
Uses StandardScaler to normalize:

Amount → scaled_amount

Time → scaled_time

Drops original Amount and Time columns.

12. Final Dataset Check
Displays cleaned dataset shape after processing.

Output
EDA Results: Class imbalance plots, correlation heatmap, transaction amount distribution.

Clean Dataset: Ready for modeling.


