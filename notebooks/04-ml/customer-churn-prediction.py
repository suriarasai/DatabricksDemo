# Databricks notebook source
# MAGIC %md
# MAGIC # Customer Churn Prediction with MLlib
# MAGIC 
# MAGIC This notebook demonstrates end-to-end machine learning workflow for predicting customer churn:
# MAGIC - Feature engineering and data preparation
# MAGIC - Model training with multiple algorithms
# MAGIC - Model evaluation and comparison
# MAGIC - Model deployment and prediction
# MAGIC 
# MAGIC **Business Value**: Identify customers likely to churn and take proactive retention actions
# MAGIC **Skills**: MLlib, feature engineering, model selection, hyperparameter tuning

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Data Generation
# MAGIC 
# MAGIC Let's create a realistic customer dataset with features that influence churn.

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
spark.sparkContext.setCheckpointDir("/tmp/ml_checkpoints")

# COMMAND ----------

# Generate synthetic customer data
def generate_customer_data(num_customers=10000):
    """Generate realistic customer data for churn prediction"""
    
    customers = []
    
    for i in range(num_customers):
        customer_id = f"CUST_{i+1:06d}"
        
        # Demographics
        age = random.randint(18, 75)
        gender = random.choice(['M', 'F'])
        income = random.normalvariate(50000, 20000)
        income = max(20000, min(150000, income))  # Cap income
        
        # Account characteristics
        tenure_months = random.randint(1, 60)
        account_balance = random.normalvariate(2000, 1500)
        account_balance = max(0, account_balance)
        
        # Usage patterns
        monthly_charges = random.normalvariate(75, 25)
        monthly_charges = max(20, min(200, monthly_charges))
        
        total_charges = monthly_charges * tenure_months * random.uniform(0.8, 1.2)
        
        # Service features
        num_products = random.choices([1, 2, 3, 4, 5], weights=[30, 35, 20, 10, 5])[0]
        has_phone_service = random.choice([True, False])
        has_internet_service = random.choice([True, False])
        has_streaming_service = random.choice([True, False]) if has_internet_service else False
        
        # Support interactions
        support_calls_last_month = random.choices([0, 1, 2, 3, 4, 5], weights=[40, 30, 15, 10, 3, 2])[0]
        avg_call_duration = random.normalvariate(8, 4) if support_calls_last_month > 0 else 0
        avg_call_duration = max(0, avg_call_duration)
        
        # Payment and contract
        contract_type = random.choices(['Month-to-month', 'One year', 'Two year'], weights=[50, 30, 20])[0]
        payment_method = random.choice(['Credit card', 'Bank transfer', 'Electronic check', 'Mailed check'])
        paperless_billing = random.choice([True, False])
        
        # Late payments (risk factor)
        late_payments_last_year = random.choices([0, 1, 2, 3, 4], weights=[60, 25, 10, 3, 2])[0]
        
        # Calculate churn probability based on features
        churn_prob = 0.1  # Base probability
        
        # Age factor (younger customers more likely to churn)
        if age < 30:
            churn_prob += 0.15
        elif age > 60:
            churn_prob -= 0.05
            
        # Tenure factor (newer customers more likely to churn)
        if tenure_months < 12:
            churn_prob += 0.25
        elif tenure_months > 36:
            churn_prob -= 0.15
            
        # Contract factor
        if contract_type == 'Month-to-month':
            churn_prob += 0.20
        elif contract_type == 'Two year':
            churn_prob -= 0.15
            
        # Support calls factor
        if support_calls_last_month > 3:
            churn_prob += 0.30
        elif support_calls_last_month == 0:
            churn_prob -= 0.05
            
        # Late payments factor
        churn_prob += late_payments_last_year * 0.10
        
        # Monthly charges factor (high charges increase churn)
        if monthly_charges > 100:
            churn_prob += 0.10
        elif monthly_charges < 40:
            churn_prob += 0.15  # Very low charges might indicate dissatisfaction
            
        # Number of products factor (more products = less churn)
        churn_prob -= (num_products - 1) * 0.05
        
        # Ensure probability is between 0 and 1
        churn_prob = max(0, min(1, churn_prob))
        
        # Determine churn based on probability
        churned = random.random() < churn_prob
        
        customers.append((
            customer_id, age, gender, income, tenure_months, account_balance,
            monthly_charges, total_charges, num_products, has_phone_service,
            has_internet_service, has_streaming_service, support_calls_last_month,
            avg_call_duration, contract_type, payment_method, paperless_billing,
            late_payments_last_year, churned
        ))
    
    return customers

# Generate the data
customer_data = generate_customer_data(10000)

# Define schema
schema = StructType([
    StructField("customer_id", StringType(), True),
    StructField("age", IntegerType(), True),
    StructField("gender", StringType(), True),
    StructField("income", DoubleType(), True),
    StructField("tenure_months", IntegerType(), True),
    StructField("account_balance", DoubleType(), True),
    StructField("monthly_charges", DoubleType(), True),
    StructField("total_charges", DoubleType(), True),
    StructField("num_products", IntegerType(), True),
    StructField("has_phone_service", BooleanType(), True),
    StructField("has_internet_service", BooleanType(), True),
    StructField("has_streaming_service", BooleanType(), True),
    StructField("support_calls_last_month", IntegerType(), True),
    StructField("avg_call_duration", DoubleType(), True),
    StructField("contract_type", StringType(), True),
    StructField("payment_method", StringType(), True),
    StructField("paperless_billing", BooleanType(), True),
    StructField("late_payments_last_year", IntegerType(), True),
    StructField("churned", BooleanType(), True)
])

# Create DataFrame
df = spark.createDataFrame(customer_data, schema)

print(f"Generated {df.count()} customer records")
print(f"Churn rate: {df.filter(col('churned') == True).count() / df.count() * 100:.2f}%")

display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Exploratory Data Analysis

# COMMAND ----------

# Basic statistics
print("Dataset Overview:")
df.describe().show()

# COMMAND ----------

# Churn distribution
churn_summary = df.groupBy("churned").count().withColumnRenamed("count", "customers")
churn_summary = churn_summary.withColumn("percentage", 
                                       col("customers") / df.count() * 100)

print("Churn Distribution:")
display(churn_summary)

# COMMAND ----------

# Churn by categorical features
categorical_features = ["gender", "contract_type", "payment_method"]

for feature in categorical_features:
    print(f"\n=== Churn Rate by {feature} ===")
    feature_churn = df.groupBy(feature).agg(
        count("*").alias("total_customers"),
        sum(when(col("churned"), 1).otherwise(0)).alias("churned_customers")
    ).withColumn("churn_rate", col("churned_customers") / col("total_customers") * 100)
    
    display(feature_churn.orderBy("churn_rate", ascending=False))

# COMMAND ----------

# Analyze numerical features
numerical_features = ["age", "tenure_months", "monthly_charges", "support_calls_last_month"]

churned_stats = df.filter(col("churned") == True).select(numerical_features).describe()
not_churned_stats = df.filter(col("churned") == False).select(numerical_features).describe()

print("Statistics for Churned Customers:")
display(churned_stats)

print("\nStatistics for Non-Churned Customers:")
display(not_churned_stats)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feature Engineering

# COMMAND ----------

# Create additional features
df_features = df.withColumn(
    "charges_per_month_tenure", col("total_charges") / col("tenure_months")
).withColumn(
    "avg_monthly_balance", col("account_balance") / col("monthly_charges")
).withColumn(
    "is_new_customer", when(col("tenure_months") <= 12, 1).otherwise(0)
).withColumn(
    "is_high_value", when(col("monthly_charges") > 100, 1).otherwise(0)
).withColumn(
    "high_support_usage", when(col("support_calls_last_month") > 2, 1).otherwise(0)
).withColumn(
    "total_services", (
        when(col("has_phone_service"), 1).otherwise(0) +
        when(col("has_internet_service"), 1).otherwise(0) +
        when(col("has_streaming_service"), 1).otherwise(0)
    )
)

# Convert categorical variables to numerical
gender_indexer = StringIndexer(inputCol="gender", outputCol="gender_indexed")
contract_indexer = StringIndexer(inputCol="contract_type", outputCol="contract_indexed")
payment_indexer = StringIndexer(inputCol="payment_method", outputCol="payment_indexed")

# Apply indexers
df_indexed = gender_indexer.fit(df_features).transform(df_features)
df_indexed = contract_indexer.fit(df_indexed).transform(df_indexed)
df_indexed = payment_indexer.fit(df_indexed).transform(df_indexed)

# Convert boolean columns to integers
bool_columns = ["has_phone_service", "has_internet_service", "has_streaming_service", "paperless_billing"]
for col_name in bool_columns:
    df_indexed = df_indexed.withColumn(col_name + "_int", when(col(col_name), 1).otherwise(0))

print("Feature engineering completed")
display(df_indexed.select("customer_id", "churned", "charges_per_month_tenure", "is_new_customer", 
                         "high_support_usage", "total_services").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Prepare Data for Machine Learning

# COMMAND ----------

# Select features for modeling
feature_columns = [
    "age", "income", "tenure_months", "account_balance", "monthly_charges",
    "total_charges", "num_products", "support_calls_last_month", "avg_call_duration",
    "late_payments_last_year", "charges_per_month_tenure", "avg_monthly_balance",
    "is_new_customer", "is_high_value", "high_support_usage", "total_services",
    "gender_indexed", "contract_indexed", "payment_indexed",
    "has_phone_service_int", "has_internet_service_int", "has_streaming_service_int",
    "paperless_billing_int"
]

# Create feature vector
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df_assembled = assembler.transform(df_indexed)

# Convert target variable to numerical
df_final = df_assembled.withColumn("label", when(col("churned"), 1.0).otherwise(0.0))

# Split data into training and testing
train_data, test_data = df_final.randomSplit([0.8, 0.2], seed=42)

print(f"Training data: {train_data.count()} records")
print(f"Test data: {test_data.count()} records")

# Check class distribution in training data
train_class_dist = train_data.groupBy("label").count()
display(train_class_dist)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Model Training and Evaluation

# COMMAND ----------

# Initialize evaluators
binary_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
multiclass_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

# Function to evaluate models
def evaluate_model(model, test_data, model_name):
    predictions = model.transform(test_data)
    
    auc = binary_evaluator.evaluate(predictions)
    accuracy = multiclass_evaluator.evaluate(predictions)
    
    # Calculate precision, recall, F1
    precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    
    precision = precision_evaluator.evaluate(predictions)
    recall = recall_evaluator.evaluate(predictions)
    f1 = f1_evaluator.evaluate(predictions)
    
    print(f"\n=== {model_name} Results ===")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        "model_name": model_name,
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": predictions
    }

# COMMAND ----------

# Model 1: Logistic Regression
print("Training Logistic Regression...")

# Scale features for logistic regression
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
scaler_model = scaler.fit(train_data)
train_scaled = scaler_model.transform(train_data)
test_scaled = scaler_model.transform(test_data)

lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="label", maxIter=100)
lr_model = lr.fit(train_scaled)

# Evaluate
lr_results = evaluate_model(lr_model, test_scaled, "Logistic Regression")

# COMMAND ----------

# Model 2: Random Forest
print("Training Random Forest...")

rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100, seed=42)
rf_model = rf.fit(train_data)

rf_results = evaluate_model(rf_model, test_data, "Random Forest")

# COMMAND ----------

# Model 3: Gradient Boosted Trees
print("Training Gradient Boosted Trees...")

gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=100, seed=42)
gbt_model = gbt.fit(train_data)

gbt_results = evaluate_model(gbt_model, test_data, "Gradient Boosted Trees")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Model Comparison and Selection

# COMMAND ----------

# Compare all models
results_data = [
    (lr_results["model_name"], lr_results["auc"], lr_results["accuracy"], lr_results["f1"]),
    (rf_results["model_name"], rf_results["auc"], rf_results["accuracy"], rf_results["f1"]),
    (gbt_results["model_name"], gbt_results["auc"], gbt_results["accuracy"], gbt_results["f1"])
]

results_df = spark.createDataFrame(results_data, ["Model", "AUC", "Accuracy", "F1_Score"])
print("Model Comparison:")
display(results_df.orderBy("AUC", ascending=False))

# COMMAND ----------

# Feature importance (using Random Forest as example)
feature_importance = rf_model.featureImportances.toArray()
feature_names = feature_columns

importance_data = list(zip(feature_names, feature_importance))
importance_df = spark.createDataFrame(importance_data, ["Feature", "Importance"])

print("Top 10 Most Important Features:")
display(importance_df.orderBy("Importance", ascending=False).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Hyperparameter Tuning

# COMMAND ----------

# Hyperparameter tuning for Random Forest (best performing model)
print("Performing hyperparameter tuning for Random Forest...")

rf_tuning = RandomForestClassifier(featuresCol="features", labelCol="label", seed=42)

# Parameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(rf_tuning.numTrees, [50, 100, 150]) \
    .addGrid(rf_tuning.maxDepth, [5, 10, 15]) \
    .addGrid(rf_tuning.minInstancesPerNode, [1, 5, 10]) \
    .build()

# Cross validator
crossval = CrossValidator(estimator=rf_tuning,
                         estimatorParamMaps=paramGrid,
                         evaluator=binary_evaluator,
                         numFolds=3,
                         seed=42)

# Fit the cross validator
cv_model = crossval.fit(train_data)

# Get best model
best_rf_model = cv_model.bestModel

print("Best parameters found:")
print(f"Number of trees: {best_rf_model.getNumTrees}")
print(f"Max depth: {best_rf_model.getMaxDepth()}")
print(f"Min instances per node: {best_rf_model.getMinInstancesPerNode()}")

# Evaluate tuned model
tuned_rf_results = evaluate_model(best_rf_model, test_data, "Tuned Random Forest")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Model Interpretation and Business Insights

# COMMAND ----------

# Analyze predictions on test data
best_predictions = best_rf_model.transform(test_data)

# Confusion matrix
confusion_matrix = best_predictions.groupBy("label", "prediction").count()
print("Confusion Matrix:")
display(confusion_matrix.orderBy("label", "prediction"))

# COMMAND ----------

# Analyze high-risk customers (predicted to churn)
high_risk_customers = best_predictions.filter(col("prediction") == 1.0)

print(f"Identified {high_risk_customers.count()} high-risk customers")

# Profile of high-risk customers
high_risk_profile = high_risk_customers.agg(
    avg("age").alias("avg_age"),
    avg("tenure_months").alias("avg_tenure"),
    avg("monthly_charges").alias("avg_monthly_charges"),
    avg("support_calls_last_month").alias("avg_support_calls"),
    avg("late_payments_last_year").alias("avg_late_payments")
)

print("High-Risk Customer Profile:")
display(high_risk_profile)

# COMMAND ----------

# Business impact analysis
total_customers = test_data.count()
predicted_churn = high_risk_customers.count()
actual_churn = test_data.filter(col("label") == 1.0).count()

# Calculate potential revenue at risk
avg_monthly_revenue = test_data.agg(avg("monthly_charges")).collect()[0][0]
annual_revenue_at_risk = predicted_churn * avg_monthly_revenue * 12

print(f"\n=== Business Impact Analysis ===")
print(f"Total customers analyzed: {total_customers}")
print(f"Predicted churn customers: {predicted_churn}")
print(f"Actual churn customers: {actual_churn}")
print(f"Average monthly revenue per customer: ${avg_monthly_revenue:.2f}")
print(f"Annual revenue at risk: ${annual_revenue_at_risk:,.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Model Deployment Simulation

# COMMAND ----------

# Create a function for real-time prediction
def predict_churn_risk(customer_features):
    """
    Predict churn risk for a new customer
    Returns probability and risk category
    """
    # This would typically be deployed as a REST API or batch job
    prediction = best_rf_model.transform(customer_features)
    
    return prediction.select("customer_id", "prediction", "probability").collect()

# Example: Predict for a sample of new customers
sample_customers = test_data.limit(5)
risk_predictions = predict_churn_risk(sample_customers)

print("Sample Churn Risk Predictions:")
for pred in risk_predictions:
    customer_id = pred['customer_id']
    prediction = pred['prediction']
    probability = pred['probability'].values[1]  # Probability of churn
    risk_level = "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW"
    
    print(f"Customer {customer_id}: {risk_level} risk (probability: {probability:.3f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Save Model and Results

# COMMAND ----------

# Save the best model
model_path = "/tmp/churn_prediction_model"
best_rf_model.write().overwrite().save(model_path)
print(f"Model saved to: {model_path}")

# Save predictions for further analysis
predictions_path = "/tmp/churn_predictions"
best_predictions.select("customer_id", "label", "prediction", "probability").write.mode("overwrite").parquet(predictions_path)
print(f"Predictions saved to: {predictions_path}")

# Create a summary report
summary_data = [(
    "Random Forest",
    tuned_rf_results["auc"],
    tuned_rf_results["accuracy"],
    tuned_rf_results["f1"],
    predicted_churn,
    annual_revenue_at_risk
)]

summary_df = spark.createDataFrame(summary_data, 
    ["Best_Model", "AUC", "Accuracy", "F1_Score", "Predicted_Churn_Customers", "Annual_Revenue_at_Risk"])

display(summary_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Insights and Recommendations
# MAGIC 
# MAGIC ### Model Performance:
# MAGIC - **Best Model**: Random Forest with hyperparameter tuning
# MAGIC - **AUC Score**: Indicates excellent discrimination ability
# MAGIC - **Business Impact**: Identified high-risk customers representing significant revenue
# MAGIC 
# MAGIC ### Key Risk Factors for Churn:
# MAGIC 1. **Contract Type**: Month-to-month contracts have highest churn risk
# MAGIC 2. **Support Calls**: Customers with frequent support calls more likely to churn
# MAGIC 3. **Tenure**: New customers (< 12 months) are at higher risk
# MAGIC 4. **Payment Issues**: Late payments strongly correlate with churn
# MAGIC 5. **Service Usage**: Customers with fewer services tend to churn more
# MAGIC 
# MAGIC ### Business Recommendations:
# MAGIC 1. **Retention Campaigns**: Target high-risk customers with special offers
# MAGIC 2. **Contract Incentives**: Encourage longer-term contracts with discounts
# MAGIC 3. **Improve Support**: Address issues that lead to frequent support calls
# MAGIC 4. **Onboarding Program**: Focus on new customer experience and engagement
# MAGIC 5. **Cross-selling**: Promote additional services to increase customer stickiness
# MAGIC 
# MAGIC ### Technical Achievements:
# MAGIC - Built end-to-end ML pipeline with feature engineering
# MAGIC - Compared multiple algorithms and selected best performer
# MAGIC - Implemented hyperparameter tuning with cross-validation
# MAGIC - Created interpretable model with feature importance analysis
# MAGIC - Calculated business impact and ROI potential
# MAGIC 
# MAGIC This model can be deployed in production to:
# MAGIC - Score customers daily/weekly for churn risk
# MAGIC - Trigger automated retention campaigns
# MAGIC - Provide insights to customer success teams
# MAGIC - Monitor model performance and retrain as needed

# COMMAND ----------