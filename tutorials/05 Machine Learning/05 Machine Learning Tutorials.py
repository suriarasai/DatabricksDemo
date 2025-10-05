# Databricks notebook source
# MAGIC %md
# MAGIC # Tutorial 5: Machine Learning in Databricks
# MAGIC
# MAGIC ## Overview
# MAGIC This notebook covers machine learning fundamentals:
# MAGIC - ML Basics with MLlib and scikit-learn
# MAGIC - Classification models for prediction
# MAGIC - Clustering for customer segmentation
# MAGIC - Model evaluation and validation
# MAGIC
# MAGIC **Libraries:**
# MAGIC - **MLlib**: Spark's distributed ML library
# MAGIC - **scikit-learn**: Traditional ML library (works well with Pandas)
# MAGIC - **Features**: Feature engineering, scaling, encoding

# COMMAND ----------

# Import libraries
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

print("Libraries imported successfully!")

# COMMAND ----------

# Load datasets
customer_df = spark.read.csv("/Volumes/workspace/sample/datasets/customer_data.csv", header=True, inferSchema=True)
products_df = spark.read.csv("/Volumes/workspace/sample/datasets/products.csv", header=True, inferSchema=True)
sales_df = spark.read.csv("/Volumes/workspace/sample/datasets/sales_data.csv", header=True, inferSchema=True)

print("Datasets loaded!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. ML Basics - Data Preparation
# MAGIC
# MAGIC ### Feature Engineering
# MAGIC
# MAGIC **Key Concepts:**
# MAGIC - **Features**: Input variables for ML models
# MAGIC - **Labels**: Target variable we want to predict
# MAGIC - **Feature Engineering**: Creating useful features from raw data
# MAGIC - **Feature Scaling**: Normalizing feature ranges

# COMMAND ----------

# Prepare customer data for ML
print("=== PREPARING CUSTOMER DATA ===")

# Create features for modeling
customer_ml = customer_df \
    .withColumn("age", F.col("age").cast("integer")) \
    .withColumn("annual_income", F.col("annual_income").cast("double")) \
    .withColumn("email_subscribed", F.col("email_subscribed").cast("integer")) \
    .withColumn("registration_date", F.to_date(F.col("registration_date"))) \
    .withColumn("customer_tenure_days", F.datediff(F.current_date(), F.col("registration_date"))) \
    .na.drop()

# Show sample
display(customer_ml.select("customer_id", "age", "annual_income", "email_subscribed", "customer_tenure_days", "segment").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Encoding Categorical Variables
# MAGIC
# MAGIC **Methods:**
# MAGIC - **Label Encoding**: Convert categories to numbers
# MAGIC - **One-Hot Encoding**: Create binary columns for each category
# MAGIC - **StringIndexer**: Spark's label encoder

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   *,
# MAGIC   DENSE_RANK() OVER (ORDER BY state) - 1 AS state_index
# MAGIC FROM samples.bakehouse.sales_customers

# COMMAND ----------

# Encode categorical variables
# from pyspark.ml.feature import StringIndexer

# Index the segment column (convert to numeric)
# SYNTAX: StringIndexer(inputCol="categorical_column", outputCol="indexed_column")
#indexer = StringIndexer(inputCol="segment", outputCol="segment_index")
#customer_ml = indexer.fit(customer_ml).transform(customer_ml)

# Index the state column
#state_indexer = StringIndexer(inputCol="state", outputCol="state_index")
#customer_ml = state_indexer.fit(customer_ml).transform(customer_ml)
#display(customer_ml.select("segment", "segment_index", "state", "state_index").limit(10))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Vectors
# MAGIC
# MAGIC **Concept:** ML models require features in vector format
# MAGIC
# MAGIC **VectorAssembler:**
# MAGIC - Combines multiple columns into a single vector column
# MAGIC - Required for Spark ML models

# COMMAND ----------

# Create feature vector
# SYNTAX: VectorAssembler(inputCols=[list of features], outputCol="features")
feature_cols = ["age", "annual_income", "email_subscribed", "customer_tenure_days", "state_index"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
customer_ml = assembler.transform(customer_ml)

# Show feature vectors
display(customer_ml.select("customer_id", "features", "segment").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Scaling
# MAGIC
# MAGIC **Why Scale?**
# MAGIC - Many ML algorithms are sensitive to feature scales
# MAGIC - Features with large ranges can dominate
# MAGIC - Standardization ensures all features contribute equally
# MAGIC
# MAGIC **StandardScaler:**
# MAGIC - Mean = 0, Standard Deviation = 1
# MAGIC - Formula: (value - mean) / std_dev

# COMMAND ----------

# Scale features
# SYNTAX: StandardScaler(inputCol="features", outputCol="scaled_features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
scaler_model = scaler.fit(customer_ml)
customer_ml = scaler_model.transform(customer_ml)

print("Features scaled successfully!")
display(customer_ml.select("features", "scaled_features").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Classification Models
# MAGIC
# MAGIC ### Problem: Predict Customer Segment
# MAGIC
# MAGIC **Classification:** Predict categorical outcomes
# MAGIC
# MAGIC **Use Case:** Predict customer segment based on demographics and behavior

# COMMAND ----------

# Prepare data for classification
# Split data into training and testing sets
# SYNTAX: randomSplit([training_ratio, test_ratio], seed)
train_data, test_data = customer_ml.randomSplit([0.8, 0.2], seed=42)

print(f"Training set: {train_data.count()} records")
print(f"Test set: {test_data.count()} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Logistic Regression
# MAGIC
# MAGIC **Algorithm:** Linear model for classification
# MAGIC - Simple and interpretable
# MAGIC - Works well for linearly separable data
# MAGIC - Provides probability estimates

# COMMAND ----------

# Train Logistic Regression model
# SYNTAX: LogisticRegression(featuresCol="features", labelCol="target", maxIter=iterations)
lr = LogisticRegression(
    featuresCol="scaled_features",
    labelCol="segment_index",
    maxIter=100,
    regParam=0.01
)

print("Training Logistic Regression model...")
lr_model = lr.fit(train_data)
print("Model trained!")

# Make predictions
lr_predictions = lr_model.transform(test_data)

# Show predictions
display(lr_predictions.select("customer_id", "segment", "segment_index", "prediction", "probability").limit(10))

# COMMAND ----------

# Evaluate Logistic Regression
evaluator = MulticlassClassificationEvaluator(
    labelCol="segment_index",
    predictionCol="prediction",
    metricName="accuracy"
)

lr_accuracy = evaluator.evaluate(lr_predictions)
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")

# F1 Score
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="segment_index",
    predictionCol="prediction",
    metricName="f1"
)
lr_f1 = f1_evaluator.evaluate(lr_predictions)
print(f"Logistic Regression F1 Score: {lr_f1:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random Forest Classifier
# MAGIC
# MAGIC **Algorithm:** Ensemble of decision trees
# MAGIC - Handles non-linear relationships
# MAGIC - Less prone to overfitting
# MAGIC - Provides feature importance

# COMMAND ----------

# Train Random Forest model
# SYNTAX: RandomForestClassifier(featuresCol, labelCol, numTrees)
rf = RandomForestClassifier(
    featuresCol="scaled_features",
    labelCol="segment_index",
    numTrees=100,
    maxDepth=5,
    seed=42
)

print("Training Random Forest model...")
rf_model = rf.fit(train_data)
print("Model trained!")

# Make predictions
rf_predictions = rf_model.transform(test_data)

# Evaluate
rf_accuracy = evaluator.evaluate(rf_predictions)
rf_f1 = f1_evaluator.evaluate(rf_predictions)

print(f"\nRandom Forest Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
print(f"Random Forest F1 Score: {rf_f1:.4f}")

# COMMAND ----------

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.featureImportances.toArray()
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Confusion Matrix
# MAGIC
# MAGIC **Purpose:** Visualize classification performance
# MAGIC - Rows: Actual labels
# MAGIC - Columns: Predicted labels
# MAGIC - Diagonal: Correct predictions

# COMMAND ----------

# Create confusion matrix
predictions_pd = rf_predictions.select("segment_index", "prediction").toPandas()

from sklearn.metrics import confusion_matrix, classification_report

# Get unique segments
segments = customer_ml.select("segment_index", "segment").distinct().orderBy("segment_index").collect()
segment_names = [row["segment"] for row in segments]

# Confusion matrix
cm = confusion_matrix(predictions_pd["segment_index"], predictions_pd["prediction"])

# Visualize
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=segment_names, yticklabels=segment_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest Classifier')
plt.tight_layout()
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(predictions_pd["segment_index"], predictions_pd["prediction"], 
                          target_names=segment_names))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Regression Models
# MAGIC
# MAGIC ### Problem: Predict Customer Annual Income
# MAGIC
# MAGIC **Regression:** Predict continuous numerical values

# COMMAND ----------

# Prepare data for regression
# Use different features to predict income
regression_features = ["age", "customer_tenure_days", "email_subscribed", "state_index"]

reg_assembler = VectorAssembler(inputCols=regression_features, outputCol="reg_features")
customer_reg = reg_assembler.transform(customer_ml)

# Scale features
reg_scaler = StandardScaler(inputCol="reg_features", outputCol="scaled_reg_features", withMean=True, withStd=True)
customer_reg = reg_scaler.fit(customer_reg).transform(customer_reg)

# Split data
train_reg, test_reg = customer_reg.randomSplit([0.8, 0.2], seed=42)

print(f"Training set: {train_reg.count()} records")
print(f"Test set: {test_reg.count()} records")

# COMMAND ----------

# Train Linear Regression model
# SYNTAX: LinearRegression(featuresCol, labelCol)
lin_reg = LinearRegression(
    featuresCol="scaled_reg_features",
    labelCol="annual_income",
    maxIter=100,
    regParam=0.1
)

print("Training Linear Regression model...")
lin_reg_model = lin_reg.fit(train_reg)
print("Model trained!")

# Make predictions
lin_reg_predictions = lin_reg_model.transform(test_reg)

# Evaluate
reg_evaluator = RegressionEvaluator(labelCol="annual_income", predictionCol="prediction")

rmse = reg_evaluator.evaluate(lin_reg_predictions, {reg_evaluator.metricName: "rmse"})
r2 = reg_evaluator.evaluate(lin_reg_predictions, {reg_evaluator.metricName: "r2"})
mae = reg_evaluator.evaluate(lin_reg_predictions, {reg_evaluator.metricName: "mae"})

print(f"\nLinear Regression Performance:")
print(f"RMSE: ${rmse:,.2f}")
print(f"MAE: ${mae:,.2f}")
print(f"R² Score: {r2:.4f}")

# COMMAND ----------

# Visualize predictions vs actual
pred_vs_actual = lin_reg_predictions.select("annual_income", "prediction").toPandas()

plt.figure(figsize=(10, 6))
plt.scatter(pred_vs_actual["annual_income"], pred_vs_actual["prediction"], alpha=0.5)
plt.plot([pred_vs_actual["annual_income"].min(), pred_vs_actual["annual_income"].max()],
         [pred_vs_actual["annual_income"].min(), pred_vs_actual["annual_income"].max()],
         'r--', lw=2)
plt.xlabel('Actual Income ($)')
plt.ylabel('Predicted Income ($)')
plt.title('Linear Regression: Predicted vs Actual Income')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Clustering (Unsupervised Learning)
# MAGIC
# MAGIC ### Problem: Customer Segmentation
# MAGIC
# MAGIC **Clustering:** Group similar data points without labels
# MAGIC
# MAGIC **K-Means Algorithm:**
# MAGIC - Partitions data into K clusters
# MAGIC - Minimizes within-cluster variance
# MAGIC - Requires specifying number of clusters

# COMMAND ----------

# Prepare data for clustering
clustering_features = ["age", "annual_income", "customer_tenure_days"]

cluster_assembler = VectorAssembler(inputCols=clustering_features, outputCol="cluster_features")
customer_cluster = cluster_assembler.transform(customer_ml)

# Scale features (important for K-Means)
cluster_scaler = StandardScaler(inputCol="cluster_features", outputCol="scaled_cluster_features", 
                                withMean=True, withStd=True)
customer_cluster = cluster_scaler.fit(customer_cluster).transform(customer_cluster)

print("Data prepared for clustering!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Elbow Method - Finding Optimal K
# MAGIC
# MAGIC **Purpose:** Determine the best number of clusters

# COMMAND ----------

# Find optimal number of clusters using elbow method
costs = []
K_range = range(2, 11)

print("Testing different K values...")
for k in K_range:
    kmeans = KMeans(featuresCol="scaled_cluster_features", k=k, seed=42)
    model = kmeans.fit(customer_cluster)
    cost = model.summary.trainingCost
    costs.append(cost)
    print(f"K={k}: Cost={cost:.2f}")

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, costs, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares')
plt.title('Elbow Method for Optimal K')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# Train K-Means with optimal K (let's use K=4)
# SYNTAX: KMeans(featuresCol, k, seed)
kmeans = KMeans(featuresCol="scaled_cluster_features", k=4, seed=42, maxIter=100)

print("Training K-Means model with K=4...")
kmeans_model = kmeans.fit(customer_cluster)
print("Model trained!")

# Make predictions (assign clusters)
customer_clustered = kmeans_model.transform(customer_cluster)

# Show cluster assignments
display(customer_clustered.select("customer_id", "age", "annual_income", "customer_tenure_days", "prediction").limit(20))

# COMMAND ----------

# Analyze clusters
cluster_summary = customer_clustered.groupBy("prediction").agg(
    F.count("customer_id").alias("size"),
    F.avg("age").alias("avg_age"),
    F.avg("annual_income").alias("avg_income"),
    F.avg("customer_tenure_days").alias("avg_tenure_days"),
    F.min("annual_income").alias("min_income"),
    F.max("annual_income").alias("max_income")
).orderBy("prediction")

display(cluster_summary)

# COMMAND ----------

# Visualize clusters (2D projection)
cluster_viz = customer_clustered.select("age", "annual_income", "prediction").toPandas()

plt.figure(figsize=(12, 6))

# Age vs Income
plt.subplot(1, 2, 1)
scatter = plt.scatter(cluster_viz["age"], cluster_viz["annual_income"], 
                     c=cluster_viz["prediction"], cmap='viridis', alpha=0.6, s=50)
plt.xlabel('Age')
plt.ylabel('Annual Income ($)')
plt.title('Customer Clusters: Age vs Income')
plt.colorbar(scatter, label='Cluster')

# Cluster distribution
plt.subplot(1, 2, 2)
cluster_counts = cluster_viz["prediction"].value_counts().sort_index()
plt.bar(cluster_counts.index, cluster_counts.values, color='skyblue')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.title('Cluster Size Distribution')

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cluster Profiling
# MAGIC
# MAGIC **Purpose:** Understand characteristics of each cluster

# COMMAND ----------

# Create cluster profiles
for cluster_id in range(4):
    print(f"\n{'='*60}")
    print(f"CLUSTER {cluster_id} PROFILE")
    print(f"{'='*60}")
    
    cluster_data = customer_clustered.filter(F.col("prediction") == cluster_id)
    
    stats = cluster_data.select(
        F.count("customer_id").alias("size"),
        F.avg("age").alias("avg_age"),
        F.avg("annual_income").alias("avg_income"),
        F.avg("customer_tenure_days").alias("avg_tenure"),
        F.sum(F.when(F.col("email_subscribed") == 1, 1).otherwise(0)).alias("subscribed")
    ).collect()[0]
    
    print(f"Size: {stats['size']} customers")
    print(f"Average Age: {stats['avg_age']:.1f} years")
    print(f"Average Income: ${stats['avg_income']:,.2f}")
    print(f"Average Tenure: {stats['avg_tenure']:.0f} days ({stats['avg_tenure']/365:.1f} years)")
    print(f"Email Subscribed: {stats['subscribed']} ({stats['subscribed']/stats['size']*100:.1f}%)")
    
    # Dominant segment in this cluster
    segment_dist = cluster_data.groupBy("segment").count().orderBy(F.desc("count")).first()
    print(f"Dominant Segment: {segment_dist['segment']} ({segment_dist['count']/stats['size']*100:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Model Evaluation Best Practices
# MAGIC
# MAGIC ### Classification Metrics
# MAGIC
# MAGIC **Key Metrics:**
# MAGIC - **Accuracy**: Overall correctness (TP+TN)/(TP+TN+FP+FN)
# MAGIC - **Precision**: TP/(TP+FP) - How many positive predictions were correct
# MAGIC - **Recall**: TP/(TP+FN) - How many actual positives were found
# MAGIC - **F1 Score**: Harmonic mean of precision and recall
# MAGIC - **ROC AUC**: Area under ROC curve

# COMMAND ----------

# Comprehensive model comparison
models_comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [lr_accuracy, rf_accuracy],
    'F1 Score': [lr_f1, rf_f1]
})

print("Model Comparison:")
print(models_comparison.to_string(index=False))

# Visualize
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison
ax[0].bar(models_comparison['Model'], models_comparison['Accuracy'], color=['skyblue', 'lightgreen'])
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Model Accuracy Comparison')
ax[0].set_ylim([0, 1])
for i, v in enumerate(models_comparison['Accuracy']):
    ax[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# F1 Score comparison
ax[1].bar(models_comparison['Model'], models_comparison['F1 Score'], color=['coral', 'gold'])
ax[1].set_ylabel('F1 Score')
ax[1].set_title('Model F1 Score Comparison')
ax[1].set_ylim([0, 1])
for i, v in enumerate(models_comparison['F1 Score']):
    ax[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cross-Validation
# MAGIC
# MAGIC **Purpose:** Assess model generalization
# MAGIC - Split data into K folds
# MAGIC - Train on K-1 folds, test on 1 fold
# MAGIC - Repeat K times
# MAGIC - Average performance across folds

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Create parameter grid
param_grid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50, 100]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .build()

# Create cross-validator
cv = CrossValidator(
    estimator=rf,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3,
    seed=42
)

print("Running cross-validation...")
print("This may take a few minutes...")

cv_model = cv.fit(train_data)

print(f"\nBest model parameters:")
print(f"Number of Trees: {cv_model.bestModel.getNumTrees}")
print(f"Max Depth: {cv_model.bestModel.getMaxDepth()}")

# Evaluate on test set
cv_predictions = cv_model.transform(test_data)
cv_accuracy = evaluator.evaluate(cv_predictions)
print(f"\nCross-Validated Model Accuracy: {cv_accuracy:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Takeaways
# MAGIC
# MAGIC **ML Workflow:**
# MAGIC 1. Data preparation and feature engineering
# MAGIC 2. Train-test split
# MAGIC 3. Model training
# MAGIC 4. Model evaluation
# MAGIC 5. Hyperparameter tuning
# MAGIC 6. Model deployment
# MAGIC
# MAGIC **Classification:**
# MAGIC - Logistic Regression: Simple, interpretable
# MAGIC - Random Forest: Powerful, handles non-linearity
# MAGIC - Evaluate with accuracy, precision, recall, F1
# MAGIC - Use confusion matrix to understand errors
# MAGIC
# MAGIC **Regression:**
# MAGIC - Predict continuous values
# MAGIC - Evaluate with RMSE, MAE, R²
# MAGIC - Visualize predictions vs actuals
# MAGIC
# MAGIC **Clustering:**
# MAGIC - K-Means for customer segmentation
# MAGIC - Use elbow method to find optimal K
# MAGIC - Profile clusters to understand segments
# MAGIC - Unsupervised learning - no labels needed
# MAGIC
# MAGIC **Model Evaluation:**
# MAGIC - Always use held-out test data
# MAGIC - Use cross-validation for robust estimates
# MAGIC - Compare multiple models
# MAGIC - Consider both performance and interpretability
# MAGIC
# MAGIC **Next Steps:** Move to Notebook 6 for Real-World Use Cases!
