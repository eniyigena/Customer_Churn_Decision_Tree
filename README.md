# Customer Churn Prediction Using Decision Trees

### Business Analytics | Machine Learning | Python | RapidMiner

## Project Overview

In this project, I applied decision tree classification to predict customer churn using historical customer data. My goal was to understand how customer behavior, account history, and service characteristics can help identify customers who are more likely to terminate their contracts.

Beyond building predictive models, I focused on evaluating their performance from a business perspective, particularly the trade-off between identifying customers at risk of leaving and unnecessarily targeting customers who are likely to stay.

I conducted the analysis primarily using **Python and RapidMiner**, comparing different decision tree configurations. I also explored model performance using **Altair** as an additional analytical tool to compare results with my Python analysis. The Altair work is not included in this repository.

## Dataset

The analysis used a course-provided dataset containing **31,891 customer records**, 11 predictor variables, and one binary target variable (`churndep`), indicating whether a customer churned.

The predictors included:

- **Customer activity:** Monthly revenue, incoming and outgoing calls.
- **Account and equipment history:** Months in service and equipment age.
- **Customer characteristics:** Marital status, international travel, computer ownership, credit card ownership, and web-capable handset.
- **Retention history:** Previous calls to the retention team.

The dataset was nearly balanced, with approximately 49.72% of customers classified as churners and 50.28% as non-churners.

## Methodology

I followed a structured machine learning workflow:

1. **Data exploration:** Examined the dataset, checked for missing values, and reviewed descriptive statistics and class distribution.
2. **Data preparation:** Separated predictors from the target variable and split the data into 75% training and 25% testing sets in Python.
3. **Model development:** Built decision tree classifiers using entropy (information gain) and Gini impurity, experimenting with different maximum tree depths.
4. **Model evaluation:** Compared accuracy, classification error, precision, recall, F1-score, and confusion matrices.
5. **Cross-tool analysis:** Conducted additional experiments in RapidMiner, including filtering and pruning, and explored results using Altair.
6. **Business interpretation:** Evaluated the models based on their ability to identify potential churners while considering the cost of unnecessary retention efforts.

## Model Performance

I compared five decision tree configurations in Python to understand how splitting criteria and tree depth affect predictive performance.

| Model | Criterion | Max Depth | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|
| Model 1 | Entropy | Unlimited | 53.74% | 53.53% | 52.63% | 53.08% |
| **Model 2** | **Entropy** | **5** | **58.66%** | **56.16%** | **75.44%** | **64.39%** |
| Model 3 | Entropy | 10 | 57.21% | 55.38% | 70.08% | 61.87% |
| Model 4 | Gini | Unlimited | 53.66% | 53.26% | 52.73% | 53.00% |
| Model 5 | Gini | 5 | 58.90% | 56.53% | 73.72% | 63.99% |

### Key Findings

**1. Tree depth had a significant impact on performance.**

The unrestricted decision trees performed relatively poorly on the test data, suggesting overfitting. Limiting tree depth improved generalization and produced more interpretable models.

**2. The highest accuracy was not necessarily the best business choice.**

Although the depth-5 Gini model achieved the highest accuracy (58.90%), I selected the depth-5 entropy model because it achieved the highest recall (75.44%) and F1-score (64.39%) among the Python models.

**3. Recall was particularly important for the business objective.**

The recommended model correctly identified approximately 75 out of every 100 customers who actually churned. However, its precision of 56.16% also indicated that some customers flagged as potential churners would not leave.

From a retention perspective, I considered this trade-off reasonable under the assumption that missing a potential churner could be more costly than contacting a customer who would otherwise stay.

**4. Cross-tool comparisons reinforced the importance of model configuration.**

My RapidMiner experiments also showed that model depth, pruning, and evaluation settings influence classification results. The analysis highlighted why performance metrics and modeling assumptions need to be considered together rather than relying on a single accuracy score.

## Business Implications

This project reinforced my understanding that **building a predictive model is only part of solving a business problem. Selecting the right model also requires understanding the decisions its predictions will support.**

For customer retention, a model with relatively high recall can help businesses identify more potentially at-risk customers and prioritize retention efforts.

However, predictive performance alone does not demonstrate financial value. Before using such a model in practice, a business would need to consider customer lifetime value, retention campaign costs, and whether interventions actually reduce churn.

## Tools and Technologies

- **Python:** pandas, NumPy, scikit-learn, Matplotlib
- **RapidMiner:** Decision tree modeling, pruning, and performance evaluation
- **Altair:** Additional exploratory analysis and comparison with Python results (not included in this repository)
- **Jupyter Notebook:** Python implementation and experimentation

## Limitations and Future Improvements

While the analysis provided useful insights into customer churn prediction, the models' predictive accuracy remained modest.

As a next step, I would explore cross-validation and hyperparameter tuning, compare decision trees with other classification algorithms such as random forests and logistic regression, and evaluate model performance using business-oriented cost metrics.

I would also use a separate validation process for model selection and reserve an untouched test set for final evaluation.

## Conclusion

The main takeaway from this project was the importance of connecting machine learning performance with business objectives.

By comparing decision tree configurations and evaluating precision, recall, and F1-score, I learned how different modeling choices influence both predictive performance and potential business decisions.

Although the models would require further improvement before practical deployment, this project helped strengthen my understanding of **classification modeling, model evaluation, overfitting, and business-driven decision-making**.

---

*This project was completed as part of my academic coursework in Business Analytics. The dataset was provided for educational purposes.*
