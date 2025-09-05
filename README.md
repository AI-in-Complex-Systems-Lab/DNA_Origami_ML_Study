# Machine Learning for DNA Origami Shape Discrimination

This project uses a Random Forest classifier and a feature engineering strategy called the "Slope Model" to classify three different DNA origami nanostructures from nanosensor array data.

## Requirements

The script requires the following Python libraries. You can install them using pip:
```bash
pip install pandas numpy seaborn matplotlib plotly scikit-learn openpyxl
```

## Data

The analysis requires the following data file to be present in the same directory as the script:
- `Fluorescence_recovery_24-32_min_updated.xlsx`

## How to Run

Execute the script from your terminal:
```bash
python final_origami_classification.py
```

## Methodology

The script performs the following key steps:
1.  **Loads and reshapes** the time-series fluorescence data.
2.  **Performs feature engineering** to create the "Slope Model" features, which capture the dynamics of the sensor response.
3.  **Compares the performance** of a model trained on the engineered features against one trained on raw data, demonstrating the superiority of the feature engineering approach.
4.  **Evaluates the final Random Forest model** using 3-fold stratified cross-validation, generating a classification report and a confusion matrix.
5.  **Visualizes the class separation** using 2D and 3D PLS-DA scores plots.

## Output

The script will:
- Print the model comparison and final performance metrics to the console.
- Display several plots, including the confusion matrix and the PLS-DA visualizations.
- Save the following files to the directory:
  - `slopes_features.xlsx`: The engineered feature set.
  - `confusion_matrix.eps`: The confusion matrix plot.
  - `plsda_2d.eps`: The 2D PLS-DA scores plot.