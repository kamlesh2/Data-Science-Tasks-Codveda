📘 Data Science Task Report
Author: Kamlesh 
Duration: 24-09-2025 to 24-10-2025 
Tools Used: Python, pandas, scikit-learn, statsmodels, TensorFlow, Keras, matplotlib, seaborn, BeautifulSoup, requests 
Data Sources: Wikipedia, local CSVs, Yahoo Finance, Fashion MNIST .ubyte files

🔹 Level 1: Basic
✅ Task 1: Web Scraping
Objective: Scrape Oscar-winning film data from Wikipedia 
Result:

Extracted 150+ films from 1964–2024

Fields: Title, Awards, Nominations, Year

Saved to oscar_wikipedia_films.csv


✅ Task 2: Data Cleaning
Objective: Clean and encode scraped data for modeling 
Result:

Final dataset shape: (1276, 4)

Standardized Awards, Nominations, and Year

Encoded Title using label encoding

Saved to cleaned_dataset.csv

✅ Task 3: Exploratory Data Analysis (EDA)
Objective: Explore relationships and distributions 
Result:

Awards distribution is right-skewed

Weak positive correlation between Awards and Nominations

Title encoding not analytically meaningful

Dataset ready for modeling

🔹 Level 2: Intermediate
✅ Task 1: Regression
Objective: Predict gold prices using regression models 
Result:

Best Model: Random Forest

RMSE: ₹2,193 | R²: 0.9863

Forecast (2026–2030): ₹104,569 each year

Linear Regression underperformed due to non-linearity

✅ Task 2: Classification
Objective: Predict wine quality using classification models 
Result:

Best Model: Random Forest

Accuracy: 90% | AUC: 0.9425

Logistic Regression: 86.6% accuracy, low recall

SVM: High precision, low recall

Class imbalance impacted recall

✅ Task 3: Clustering
Objective: Segment Netflix titles using K-Means 
Result:

Clustered into 4 groups using PCA

Cluster 0: Feature films & documentaries

Cluster 1: TV shows

Cluster 2: Action & family films

Cluster 3: Sparse/miscellaneous genres

🔹 Level 3: Advanced
✅ Task 1: Time Series Forecasting
Objective: Forecast stock prices using ARIMA 
Result:

ARIMA RMSE: 278.6

Daily frequency inferred

Model successfully forecasted future stock prices

✅ Task 2: NLP – Spam Detection
Objective: Detect spam using frequency-based features 
Result:

Naive Bayes Accuracy: 82%

Precision: 0.95 (non-spam), 0.72 (spam)

Recall: 0.73 (non-spam), 0.95 (spam)

Logistic Regression Accuracy: 92% ✅

Balanced precision and recall

Best overall performance

✅ Task 3: Neural Networks
Objective: Classify Fashion MNIST images using TensorFlow/Keras 
Result:

Final Test Accuracy: 86.67%

Validation Accuracy peaked at 87.68%

Loss steadily decreased across 15 epochs

Model generalized well without overfitting

✅ Completion Summary
Level	Task	Status
1	Web Scraping, Cleaning, EDA	✅
2	Regression, Classification, Clustering	✅
3	Time Series, NLP, Neural Networks	✅








2	Regression, Classification, Clustering	✅
3	Time Series, NLP, Neural Networks	✅
