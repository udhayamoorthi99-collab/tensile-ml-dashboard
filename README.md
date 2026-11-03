

# Tensile Test ML Dashboard

An interactive **Machine Learning dashboard** for analyzing tensile-test data and predicting the **Maximum Tensile Force (N)** of Planar and Non-Planar specimens.

###  Live Dashboard

https://tensile-ml-dashboard-nvvzgywwmchambwrsnqj3i.streamlit.app/

## Project Overview

A **Random Forest Regressor** predicts maximum tensile force using:

- Orientation Angle (0°–90°)
- Cross-Sectional Area (mm²)
- Specimen Type (Planar / Non-Planar)

The dashboard provides experimental-data visualization, model evaluation, feature importance, interactive force prediction, and dataset exploration.

## Model Performance

**Random 80/20 split:**
- R²: **0.9423**
- MAE: **79.48 N**
- RMSE: **107.64 N**

**Unseen-orientation validation:**
- R²: **0.7866**
- MAE: **141.38 N**
- RMSE: **157.67 N**

## Tech Stack

`Python` · `Scikit-learn` · `Pandas` · `Matplotlib` · `Streamlit` · `Random Forest`

## Future Work

The next stage will extend the project from maximum-force prediction to **stress-strain curve modeling and physics-guided neural networks**.

> This dashboard is an experimental research prototype and not a certified engineering design tool.
