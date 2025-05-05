# st_result-analyse

A Streamlit-based web application to automate the analysis of student academic performance. Users can upload CSV or Excel files containing student marks and generate insights, including top-ranked students, subject pass percentages, and grouped lists of students by subject failures.

## Features

**File Upload:** Drag-and-drop or browse to upload CSV/Excel files with student marks.

**Insights Generation:** One-click analysis to display:

- Top 10 ranked students by total marks with their names and scores.

- Pass percentage for each subject (configurable passing mark: 35).

- Students grouped by subject failures (1, 2, 3, or more than 3).

- Average marks per subject for class performance overview.

**Predictive Analytics:** Linear regression model to forecast student performance trends based on historical marks.


**Interactive Visualizations:**

- Bar charts for subject-wise pass percentages.

- Pie charts for distribution of students by number of failures.

- Line charts for predicted vs. actual marks.

## Project Structure

st-result-analyse/

├── st_result.py   # Main Streamlit application

├── requirements.txt    # Python dependencies

└── README.md           # Project documentation
