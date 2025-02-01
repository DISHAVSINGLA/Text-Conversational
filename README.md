# TOPSIS-Based Ranking of Conversational AI Models

## Overview
This project applies the **TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)** method to rank various **pre-trained conversational AI models** based on multiple evaluation criteria.

## Task
As per the assignment, the roll number determines the task. Since the roll number ends with **4**, the assigned task is **Text Conversational Models**.

## Models Evaluated
The following conversational AI models are compared:
1. **DialoGPT (medium)**
2. **BlenderBot 3 (3B)**
3. **GPT-3.5 Turbo**
4. **Mistral-7B (Instruct)**
5. **T5 Dialogue**

## Evaluation Criteria
Each model is evaluated based on the following five criteria:
- **Coherence** (Higher is better)
- **Fluency** (Higher is better)
- **Diversity** (Higher is better)
- **Inference Speed (ms)** (Lower is better)
- **Model Size (GB)** (Lower is better)

## Methodology
1. **Data Normalization:** All values are scaled using **Min-Max Normalization**.
2. **Weighting:** Weights assigned based on importance.
3. **Ideal Best & Worst:** The best and worst values for each criterion are determined.
4. **Separation Measures:** Euclidean distances to ideal best and worst values are calculated.
5. **TOPSIS Score Calculation:** The final ranking is computed based on relative closeness to the ideal solution.

## Code Implementation
The implementation is done in Python using the following libraries:
- **NumPy & Pandas** (for numerical calculations & data handling)
- **Matplotlib & Seaborn** (for visualizing results)
- **Scikit-learn** (for data normalization)

## Results
The models are ranked based on their **TOPSIS score**, and a bar chart visualization is generated.

## Usage
1. Run the Python script `topsis_text_conversational.py`.
2. The ranked results will be printed.
3. A bar chart visualization will be displayed.
4. The final results are also saved in `topsis_results.csv`.

## File Structure
```
|-- topsis_text_conversational.py  # Python script with TOPSIS implementation
|-- topsis_results.csv             # CSV file containing ranked models
|-- README.md                      # Project documentation
```

## Conclusion
Using TOPSIS, we ranked conversational AI models based on multiple evaluation criteria, providing an objective selection of the best-performing model.

## Authors
Dishav Singla

