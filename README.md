# Company-bankruptcy-prediction

There are man companies both startups as well as established companies that do business in their countries and around the world. They have profits, losses, capital and pay taxes as well other factors that a company does or have while operating in the market. Based on these features we need to predict whether the company will go bankrupt in the next five years. This can lead to various insights and other actions on the company by the banks or other institutions for loan, share holding, risk assesment and likewise. For this purpose, models have been trained based on the data we have and it predicts the bankruptcy of the company in the coming next five years.

<br>

## Business Values

- Lending & Credit Risk

    - To assess whether a company is a safe borrower.

    - Deciding loan approvals and credit limits.

    - Identifying risky borrowers early.

- Investment Decisions

    - Venture capital, private equity, hedge funds → predict the financial health of potential portfolio companies.

    - Investors can avoid companies with a high likelihood of bankruptcy.


- Supply Chain Risk Management

    - Large corporations can predict the stability of their suppliers or distributors.

    - If a key supplier is at risk of bankruptcy, firms can diversify or find backups.


- Auditing & Compliance

    - Auditors can use the model as a red flag tool when reviewing financial statements.

     - Regulators (like central banks) may use it to monitor systemic risk in industries.

- Corporate Strategy & Early Warning

     - Companies can run their own financials through the model to monitor health.

     - Acts as an early warning system for management to take corrective actions (restructuring, cost-cutting, refinancing).

<br>

## Data

The data has been taken from the following source:<br>
https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data

<br>
The data contains five files in arff format which are to be loaded by the library having 65 features. The data has features in the form of ratios of business financials like "net profit / total assets", "net profit / sales" etc. Some of the features are listed below:<br><br>

<pre>
X1	: net profit / total assets
X2	: total liabilities / total assets
X3	: working capital / total assets
X4	: current assets / short-term liabilities
X5	: [(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365
X6	: retained earnings / total assets
...
</pre>

<br>


## Models

classification models were trained on the data to predict bakruptcy. Data was preproessed in stages to give the final processed form as model input. The featues as input to the model were given in two ways where first was removing the correlated columns removed while in second was to keep all the data together. For this purpose three models were selected and tested upon because of the complexity of the data while other models were performing very poorly. These are:
- XGBoost
- LightGBM
- CatBoost

<br>

## Evaluation

**Model performance with correlated columns removed**

|Model|Precision|Recall|F1-Score|
|-----|---------|------|--------|
|XGBoost| 0.542 | 0.656 | 0.593 |
|LightGBM | 0.542 | 0.658 | 0.594 |
| CatBoost | 0.535 | 0.636 | 0.581 |

<br>

**Model performance with all columns**



<table>

<tr>
<th>Model</th>
<th>Precision</th>
<th>Recall</th>
<th>F1 score</th>
</tr>

<tr>
<td>XGBoost</td>
<td>0.871</td>
<td><b>0.702</b></td>
<td>0.777</td>
</tr>

<tr>
<td>LightGBM</td>
<td>0.880</td>
<td>0.697</td>
<td><b>0.778</b></td>
</tr>

<tr>
<td>CatBoost</td>
<td>0.829</td>
<td>0.680</td>
<td>0.748</td>
</tr>

</table>

<br>

##  Conclusion

Here the models with all columns are performing better in terms of metrics. Among the models, XGBoost has the best recall and good F1 score also while for LightGBM, it has high F1 score but relatively lower recall. So our choice would be the XGBoost model so that it identifies all the risky companies about to be bankrupt.
