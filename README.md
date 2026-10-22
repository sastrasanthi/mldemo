# Customer Churn – Streamlit Classification Lab

An interactive educational app for binary-classification CSV datasets. Upload a CSV, choose the target and positive class, explore the data, configure preprocessing and feature selection, train six classifiers, inspect ROC/PR curves and confusion matrices, compare eleven metrics, select the best model, and download a PDF report with charts and interpretations.

## Deploy free on Streamlit Community Cloud

1. Create a **public GitHub repository**.
2. Upload `app.py`, `requirements.txt`, and `README.md` to the repository root.
3. Visit https://share.streamlit.io/ and sign in with GitHub.
4. Click **Create app**, select the repository, branch `main`, and entrypoint `app.py`.
5. Deploy. Streamlit installs the requirements automatically.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Workflow

1. Upload a CSV (e.g. IBM Telco Customer Churn).
2. Select the binary target (`Churn`) and positive class (`Yes`).
3. Configure missing-value handling, scaling, split, feature selection, and threshold in the sidebar.
4. Click **Run all** for Logistic Regression, Decision Tree, Random Forest, KNN, SVM and Gradient Boosting.
5. Use the EDA, Evaluation metrics, Graphs, Results, and Decision & PDF tabs.
6. Generate the full PDF report and download it.

## Notes

- All preprocessing and feature selection are learned from the training partition only.
- ROC-AUC and average precision are calculated from predicted probabilities.
- The "best" model is selected by the chosen metric on the test partition. For a rigorous final estimate, validate the selected model on new data or use nested cross-validation.
- This demonstration is for binary classification only.
- Avoid uploading confidential or personally identifiable customer data to publicly hosted applications.
- Free hosting may have CPU/memory limits, sleep when idle, and other platform restrictions.
