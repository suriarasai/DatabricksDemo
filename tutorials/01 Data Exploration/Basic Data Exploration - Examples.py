# Databricks notebook source
# MAGIC %md
# MAGIC # Tutorial 1: Data Exploration & Analysis
# MAGIC
# MAGIC ## Overview
# MAGIC This tutorial demonstrates how to load, explore, and understand datasets using Databricks Free Edition. This notebook covers fundamental data exploration techniques including:
# MAGIC - Loading and exploring datasets
# MAGIC - Data cleaning and quality checks
# MAGIC - Statistical analysis and profiling
# MAGIC ### Learning Objectives
# MAGIC  - Load data from various sources
# MAGIC  - Explore data structure and characteristics
# MAGIC  - Perform basic statistical analysis
# MAGIC  - Handle common data quality issues
# MAGIC
# MAGIC **Datasets Used:**
# MAGIC - customer_data.csv
# MAGIC - products.csv
# MAGIC - sales_data.csv
# MAGIC - web_traffic.csv

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Basic Data Exploration
# MAGIC
# MAGIC ### Loading Data into Databricks
# MAGIC
# MAGIC **Key Concepts:**
# MAGIC - `spark.read.csv()`: Reads CSV files into Spark DataFrame
# MAGIC - `header=True`: First row contains column names
# MAGIC - `inferSchema=True`: Automatically detects data types
# MAGIC - `.toPandas()`: Converts Spark DataFrame to Pandas for easier manipulation

# COMMAND ----------

# Import required libraries
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import *

# Load customer data
# SYNTAX: spark.read.csv("path", header=True, inferSchema=True)
customer_df = spark.read.csv("/Volumes/workspace/sample/datasets/customer_data.csv", header=True, inferSchema=True)

# Display first few rows
# SYNTAX: .display() shows data in interactive table format
display(customer_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Understanding Your Data
# MAGIC
# MAGIC **Key Functions:**
# MAGIC - `.printSchema()`: Shows column names and data types
# MAGIC - `.count()`: Returns number of rows
# MAGIC - `.columns`: Lists all column names

# COMMAND ----------

# Check schema and structure
print("Customer Data Schema:")
customer_df.printSchema()

print(f"\nTotal Rows: {customer_df.count()}")
print(f"Total Columns: {len(customer_df.columns)}")
print(f"\nColumn Names: {customer_df.columns}")

# COMMAND ----------

# Load other datasets
products_df = spark.read.csv("/Volumes/workspace/sample/datasets/products.csv", header=True, inferSchema=True)
sales_df = spark.read.csv("/Volumes/workspace/sample/datasets/sales_data.csv", header=True, inferSchema=True)
web_traffic_df = spark.read.csv("/Volumes/workspace/sample/datasets/web_traffic.csv", header=True, inferSchema=True)

print("All datasets loaded successfully!")
print(f"Products: {products_df.count()} rows")
print(f"Sales: {sales_df.count()} rows")
print(f"Web Traffic: {web_traffic_df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quick Data Profiling
# MAGIC
# MAGIC **Key Functions:**
# MAGIC - `.describe()`: Statistical summary for numeric columns
# MAGIC - `.summary()`: Extended statistics including percentiles

# COMMAND ----------

# Statistical summary of customer data
display(customer_df.describe())

# COMMAND ----------

# More detailed summary with percentiles
display(customer_df.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Cleaning
# MAGIC
# MAGIC ### Handling Missing Values
# MAGIC
# MAGIC **Key Concepts:**
# MAGIC - Check for NULL values using `.isNull().sum()`
# MAGIC - Drop nulls with `.na.drop()`
# MAGIC - Fill nulls with `.na.fill()`
# MAGIC - Replace values with `.na.replace()`

# COMMAND ----------

# Check for missing values in customer data
from pyspark.sql.functions import col, count, when, isnan

# Count nulls for each column
# SYNTAX: F.sum(F.when(condition, 1).otherwise(0))
missing_counts = customer_df.select([
    F.sum(when(col(c).isNull(), 1).otherwise(0)).alias(c) 
    for c in customer_df.columns
])

print("Missing Values by Column:")
display(missing_counts)

# COMMAND ----------

# Handle missing values - Example strategies

# Strategy 1: Drop rows with any null values
customer_clean = customer_df.na.drop()
print(f"Rows after dropping nulls: {customer_clean.count()}")

# Strategy 2: Fill missing values with defaults
# SYNTAX: .na.fill({"column_name": default_value})
customer_filled = customer_df.na.fill({
    "phone": "Unknown",
    "email_subscribed": False,
    "annual_income": 0
})

# Strategy 3: Fill with column statistics (mean, median)
avg_age = customer_df.select(F.avg("age")).first()[0]
customer_filled = customer_df.na.fill({"age": int(avg_age)})

print("Data cleaning strategies applied!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Handling Duplicates
# MAGIC
# MAGIC **Key Functions:**
# MAGIC - `.dropDuplicates()`: Removes duplicate rows
# MAGIC - `.dropDuplicates([columns])`: Removes duplicates based on specific columns

# COMMAND ----------

# Check for duplicate customer records
print(f"Total rows: {customer_df.count()}")
print(f"Unique customer_ids: {customer_df.select('customer_id').distinct().count()}")

# Remove duplicates based on customer_id
customer_unique = customer_df.dropDuplicates(['customer_id'])
print(f"Rows after removing duplicates: {customer_unique.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Quality Checks
# MAGIC
# MAGIC **Best Practices:**
# MAGIC - Check for invalid values (negative prices, ages, etc.)
# MAGIC - Validate email formats
# MAGIC - Check date ranges
# MAGIC - Identify outliers

# COMMAND ----------

# Quality checks for customer data

# 1. Check for invalid ages
invalid_age = customer_df.filter((col("age") < 0) | (col("age") > 120))
print(f"Records with invalid age: {invalid_age.count()}")

# 2. Check annual income range
income_stats = customer_df.select(
    F.min("annual_income").alias("min_income"),
    F.max("annual_income").alias("max_income"),
    F.avg("annual_income").alias("avg_income")
)
display(income_stats)

# 3. Check for valid email domains
email_pattern = "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
valid_emails = customer_df.filter(col("email").rlike(email_pattern))
print(f"Valid email addresses: {valid_emails.count()} out of {customer_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Handling Outliers
# MAGIC
# MAGIC **Methods:**
# MAGIC - IQR (Interquartile Range) method
# MAGIC - Z-score method
# MAGIC - Visual inspection

# COMMAND ----------

# Detect outliers in annual income using IQR method

# Calculate quartiles
quantiles = customer_df.approxQuantile("annual_income", [0.25, 0.75], 0.01)
Q1, Q3 = quantiles[0], quantiles[1]
IQR = Q3 - Q1

# Define outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Q1: {Q1:,.0f}, Q3: {Q3:,.0f}, IQR: {IQR:,.0f}")
print(f"Outlier bounds: [{lower_bound:,.0f}, {upper_bound:,.0f}]")

# Filter outliers
outliers = customer_df.filter(
    (col("annual_income") < lower_bound) | (col("annual_income") > upper_bound)
)
print(f"\nOutliers detected: {outliers.count()}")

# Remove outliers
customer_no_outliers = customer_df.filter(
    (col("annual_income") >= lower_bound) & (col("annual_income") <= upper_bound)
)
print(f"Records after removing outliers: {customer_no_outliers.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Statistical Analysis
# MAGIC
# MAGIC ### Descriptive Statistics
# MAGIC
# MAGIC **Key Metrics:**
# MAGIC - Central tendency: mean, median, mode
# MAGIC - Dispersion: variance, standard deviation, range
# MAGIC - Distribution: skewness, kurtosis

# COMMAND ----------

# Comprehensive statistical analysis

# Convert to Pandas for advanced statistics
customer_pd = customer_df.select("age", "annual_income").toPandas()

print("=== AGE STATISTICS ===")
print(f"Mean: {customer_pd['age'].mean():.2f}")
print(f"Median: {customer_pd['age'].median():.2f}")
print(f"Mode: {customer_pd['age'].mode()[0]:.2f}")
print(f"Std Dev: {customer_pd['age'].std():.2f}")
print(f"Variance: {customer_pd['age'].var():.2f}")
print(f"Skewness: {customer_pd['age'].skew():.2f}")

print("\n=== INCOME STATISTICS ===")
print(f"Mean: ${customer_pd['annual_income'].mean():,.2f}")
print(f"Median: ${customer_pd['annual_income'].median():,.2f}")
print(f"Std Dev: ${customer_pd['annual_income'].std():,.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Group-wise Analysis
# MAGIC
# MAGIC **Key Functions:**
# MAGIC - `.groupBy()`: Group data by one or more columns
# MAGIC - `.agg()`: Apply aggregate functions
# MAGIC - Common aggregations: count, sum, avg, min, max

# COMMAND ----------

# Analyze customers by segment
segment_analysis = customer_df.groupBy("segment").agg(
    F.count("customer_id").alias("customer_count"),
    F.avg("age").alias("avg_age"),
    F.avg("annual_income").alias("avg_income"),
    F.sum(when(col("email_subscribed") == True, 1).otherwise(0)).alias("subscribed_count")
).orderBy(F.desc("customer_count"))

display(segment_analysis)

# COMMAND ----------

# Geographic analysis by state
state_analysis = customer_df.groupBy("state").agg(
    F.count("customer_id").alias("customer_count"),
    F.avg("annual_income").alias("avg_income")
).orderBy(F.desc("customer_count")).limit(10)

display(state_analysis)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Correlation Analysis
# MAGIC
# MAGIC **Purpose:** Understand relationships between numeric variables

# COMMAND ----------

# Calculate correlation between age and income
correlation = customer_df.stat.corr("age", "annual_income")
print(f"Correlation between Age and Income: {correlation:.3f}")

# Create correlation matrix using Pandas
numeric_cols = ["age", "annual_income"]
correlation_matrix = customer_df.select(numeric_cols).toPandas().corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Product Data Analysis

# COMMAND ----------

# Analyze product data
product_summary = products_df.agg(
    F.count("product_id").alias("total_products"),
    F.avg("price").alias("avg_price"),
    F.avg("rating").alias("avg_rating"),
    F.sum("num_reviews").alias("total_reviews")
)

display(product_summary)

# Products by category
category_analysis = products_df.groupBy("category").agg(
    F.count("product_id").alias("product_count"),
    F.avg("price").alias("avg_price"),
    F.avg("rating").alias("avg_rating")
).orderBy(F.desc("product_count"))

display(category_analysis)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sales Data Profiling

# COMMAND ----------

# Sales performance metrics
sales_summary = sales_df.agg(
    F.count("transaction_id").alias("total_transactions"),
    F.sum("total_sales").alias("total_revenue"),
    F.avg("total_sales").alias("avg_transaction_value"),
    F.avg("customer_satisfaction").alias("avg_satisfaction"),
    F.sum("quantity").alias("total_units_sold")
)

display(sales_summary)

# Regional performance
regional_analysis = sales_df.groupBy("region").agg(
    F.count("transaction_id").alias("transactions"),
    F.sum("total_sales").alias("revenue"),
    F.avg("customer_satisfaction").alias("avg_satisfaction")
).orderBy(F.desc("revenue"))

display(regional_analysis)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Takeaways
# MAGIC
# MAGIC **Data Exploration:**
# MAGIC - Always start with `.printSchema()` and `.describe()` to understand your data
# MAGIC - Check data quality early: missing values, duplicates, outliers
# MAGIC
# MAGIC **Data Cleaning:**
# MAGIC - Handle missing values appropriately (drop, fill, or impute)
# MAGIC - Validate data ranges and formats
# MAGIC - Remove or treat outliers based on business context
# MAGIC
# MAGIC **Statistical Analysis:**
# MAGIC - Use descriptive statistics to understand distributions
# MAGIC - Group-wise analysis reveals patterns across categories
# MAGIC - Correlation helps identify relationships between variables
# MAGIC
# MAGIC **Next Steps:** Move to Notebook 2 for Data Visualization techniques!
