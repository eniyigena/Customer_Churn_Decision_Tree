# Customer Churn Prediction with Decision Trees

This a business analytics project using Python and RapidMiner to predict customer contract termination and evaluate the trade-off between identifying churners and unnecessary retention outreach.

I build an Entropy-based decision tree with `max_depth=5` — **75.44% recall**, **64.39% F1-score**, and **58.66% accuracy** on the reported test split.

## Business objective

Customer churn creates a need to identify customers who may terminate their contracts. This project investigates whether customer usage, account history, equipment, and personal characteristics can support that prediction.

The analysis compares five decision tree configurations and recommends a model under the business assumption that missing an actual churner is more costly than flagging a customer who would stay. This is an academic modeling study; financial returns and the effectiveness of retention interventions were not measured.

## Project highlights

- Analyzed **31,891 customer records** with **11 predictors** and one binary target.
- Used a **75% training / 25% testing** split in Python.
- Compared **entropy and Gini impurity**, with unlimited depth and depth limits of 5 or 10.
- Evaluated accuracy, classification error, precision, recall, F1-score, and the baseline confusion matrix.
- Recommended the depth-5 entropy tree for its higher churn recall and F1-score among the Python candidates.
- Conducted a complementary workflow in RapidMiner, including data filtering and pruning experiments.

## Dataset

**Source:** `HW1_Data.csv`, we choose to use this name for simplicity.

**Target:** `churndep` — `1` indicates churn; `0` indicates retention.

| Variable | Description |
| --- | --- |
| `revenue` | Mean monthly revenue in dollars |
| `outcalls` | Mean number of outbound voice calls |
| `incalls` | Mean number of inbound voice calls |
| `months` | Months in service |
| `eqpdays` | Number of days the customer has had the current equipment |
| `webcap` | Whether the handset is web capable |
| `marryyes` | Married: 1 = Yes, 0 = No |
| `travel` | Has traveled to a non-US country: 1 = Yes, 0 = No |
| `pcown` | Owns a personal computer: 1 = Yes, 0 = No |
| `creditcd` | Possesses a credit card: 1 = Yes, 0 = No |
| `retcalls` | Number of previous calls to the retention team |
| `churndep` | Customer churn: 1 = Yes, 0 = No |

The reported class distribution was approximately **50.28% retained** and **49.72% churned**, making the dataset nearly balanced. The report records no missing values in the raw dataset.

The dataset is course-provided. Access and redistribution remain subject to the data owner's and course's permissions; this project does not grant rights to redistribute it.

## Python modeling process

### 1. Explore the data

The analysis checked missing values, examined the target distribution, and reviewed descriptive statistics to understand the scale and distribution of the variables.

### 2. Split predictors and target

The code shown in the report used the first 11 columns as predictors and the final column, `churndep`, as the target. It created:

| Partition | Records | Share |
| --- | ---: | ---: |
| Training | 23,918 | 75% |
| Testing | 7,973 | 25% |

The displayed split used `train_test_split(X, y, random_state=0)`, with the default test size and no explicit stratification argument. The displayed decision tree configurations used `random_state=42`.

### 3. Establish an unrestricted baseline

The initial model used entropy with `max_depth=None`. The report describes nearly perfect training accuracy but substantially weaker test performance and a tree too complex to interpret easily. This combination is consistent with overfitting.

### 4. Compare model configurations

Five trees were compared on the test split by changing the splitting criterion and maximum depth. The report does not document a separate validation set or cross-validation; the results therefore represent an exploratory holdout comparison rather than an independent final evaluation after model selection.

### 5. Recommend a model using business priorities

The depth-5 entropy tree was recommended because it achieved the highest recall and F1-score among the Python candidates. The depth-5 Gini tree achieved slightly higher accuracy and precision.

## Python results

The following values reproduce **Part I, Table 2** of the project report, expressed as percentages. Precision, recall, and F1 refer to churn (`1`) as the positive class.

| Model | Splitting criterion | Maximum depth | Accuracy | Error | Precision | Recall | F1-score |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Entropy | Unlimited | 53.74% | 46.26% | 53.53% | 52.63% | 53.08% |
| **2 — recommended** | **Entropy** | **5** | **58.66%** | **41.34%** | **56.16%** | **75.44%** | **64.39%** |
| 3 | Entropy | 10 | 57.21% | 42.79% | 55.38% | 70.08% | 61.87% |
| 4 | Gini impurity | Unlimited | 53.66% | 46.34% | 53.26% | 52.73% | 53.00% |
| 5 | Gini impurity | 5 | 58.90% | 41.10% | 56.53% | 73.72% | 63.99% |

Limiting depth to 5 improved the reported results for both splitting criteria compared with their unlimited-depth counterparts. Within this experiment, controlling depth had a larger effect than choosing between entropy and Gini.

### Why Model 2 was recommended

Compared with the depth-5 Gini tree, Model 2 delivered:

- **1.72 percentage points higher recall**.
- **0.40 percentage points higher F1-score**.
- **0.24 percentage points lower accuracy**.

A recall of 75.44% means the model identified approximately 75 of every 100 actual churners in the test data. A precision of 56.16% means approximately 56 of every 100 customers flagged as churners actually churned.

That trade-off supports Model 2 when identifying more at-risk customers is the priority. It also means outreach would include many customers who would stay. A real retention team would need customer value, campaign costs, and intervention outcomes to decide whether that trade-off is economically worthwhile.

### Baseline confusion matrix

I found the following counts for the **initial unrestricted entropy model in Part I, Table 1**. These are not the recommended model's confusion matrix.

| Actual outcome | Predicted retained (0) | Predicted churned (1) |
| --- | ---: | ---: |
| Retained (0) | 2,213 true negatives | 1,810 false positives |
| Churned (1) | 1,890 false negatives | 2,060 true positives |

The counts total 7,973 observations and imply **53.59% accuracy**, **46.41% classification error**, **53.23% precision**, **52.15% recall**, and **52.69% F1-score**, matching the baseline metrics in Table 1 after rounding.

**Reporting note:** Table 2 gives slightly different metrics for the same nominal unrestricted entropy configuration, including 53.74% accuracy. The report does not explain this discrepancy. The comparison table above preserves Table 2, while this confusion matrix preserves Table 1. Exact confusion-matrix counts for the recommended Python model were not provided and have not been inferred from rounded metrics.

## How performance was interpreted

| Measure | Business interpretation |
| --- | --- |
| Accuracy | Share of all customers classified correctly |
| Classification error | Share classified incorrectly; equal to 1 minus accuracy |
| Precision | Among customers flagged as churners, the share who actually churned |
| Recall | Among actual churners, the share successfully identified |
| F1-score | Harmonic mean of precision and recall |
| False positives | Customers flagged for possible outreach who did not churn |
| False negatives | Actual churners the model missed |

Accuracy alone does not capture the different consequences of missed churners and unnecessary outreach. The recommendation therefore considered recall and F1 alongside accuracy and precision.

## Complementary RapidMiner analysis

The RapidMiner workflow imported the same source dataset, assigned `churndep` as the binomial label, and filtered records to retain `revenue >= 0` and `eqpdays >= 0`. It then used split validation, a Decision Tree operator, Apply Model, and Performance (Binomial Classification).

The experiments covered information gain, Gini impurity, depth limits, pruning, and an alternative 60/40 split. The report's final RapidMiner comparison table gives:

| Model | Criterion | Maximum depth | Accuracy | Error | Precision | Recall | F1-score |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Information gain | Unlimited | 53.68% | 46.32% | 53.90% | 54.26% | 54.08% |
| 2 | Information gain | 5 | 58.94% | 41.06% | 63.92% | 42.06% | 50.74% |
| 3 | Information gain | 10 | 57.49% | 42.51% | 64.70% | 36.70% | 48.22% |
| 4 | Gini impurity | Unlimited | 52.61% | 47.39% | 52.82% | 53.59% | 53.20% |
| 5 | Gini impurity | 5 | 59.22% | 40.78% | 64.87% | 41.16% | 50.37% |

These values are preserved as reported; class conventions and full operator settings should be checked before comparing them directly with Python. The RapidMiner filtering is not shown in the displayed Python workflow, and the post-filter record count is not documented.

The report recommends Model 2 within the two depth-5 RapidMiner candidates. It has slightly higher recall and F1 than Model 5, but **does not have the highest F1 across all five RapidMiner models**: Model 1 does in the table above. The narrative also gives 58.94% accuracy for Model 3, whereas the final table gives 57.49%. This README uses the final comparison table without treating the conflicting narrative as verified.

## Why decision trees?

Decision trees express predictions through if–then rules that can be discussed with business stakeholders. They can represent nonlinear relationships and interactions among customer attributes. Limiting depth makes the resulting rules easier to inspect than a fully grown tree.

Interpretability does not establish causality. A variable used in a split may help predict churn without being a cause of churn or a useful target for intervention.

## Tools and reproduction notes

The report documents a Python notebook workflow using **pandas, NumPy, scikit-learn, and Matplotlib**, alongside **RapidMiner**. Exact Python, package, and RapidMiner versions are not recorded, so an identical environment cannot be reconstructed from the report alone.

To run the Python analysis when the project code is available:

1. Obtain an authorized copy of `HW1_Data.csv`.
2. Open the analysis notebook in a Python/Jupyter environment with the listed libraries installed.
3. Place the CSV in the notebook's working directory, as expected by the displayed `pd.read_csv("HW1_Data.csv")` call, or update that path.
4. Preserve the original column order: the displayed code selects predictors and target by position.
5. Restart the kernel and run the notebook from top to bottom.
6. Compare the outputs with the reported results, retaining the documented baseline discrepancy when interpreting differences.

The recommended model's configuration shown in the report is:

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42,
)
```

This snippet defines the estimator only; data preparation, fitting, and evaluation belong to the project analysis code. Results in this README were transcribed from the report, not independently reproduced from the dataset.

## Limitations and next steps

- **Model selection reused the test split.** Future work should select parameters within training data using validation or cross-validation, then evaluate the fixed model on an untouched test set.
- **Predictive performance remains modest.** The recommended model's 58.66% accuracy is above the dataset's approximately 50% class proportions, but the project does not establish production readiness.
- **Some reported results conflict.** The unrestricted Python baseline and parts of the RapidMiner analysis need reconciliation through a clean rerun.
- **Cross-platform comparisons are not controlled.** Filtering, pruning, class conventions, and split behavior may differ between Python and RapidMiner.
- **Data meaning and timing require verification.** Negative revenue should be investigated before assuming it is erroneous, and all predictors, particularly retention calls, must be available before the churn outcome being predicted.
- **Business impact has not been measured.** A retention pilot should evaluate whether targeting predicted churners leads to incremental retention and positive net value.

Priority improvements are to rerun the analysis in a recorded environment, export the recommended model's confusion matrix, introduce an independent model-selection design, and evaluate retention outcomes before making deployment claims.
