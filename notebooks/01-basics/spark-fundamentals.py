# Databricks notebook source
# MAGIC %md
# MAGIC # Apache Spark Fundamentals
# MAGIC 
# MAGIC This notebook introduces the core concepts of Apache Spark including:
# MAGIC - SparkContext and SparkSession
# MAGIC - DataFrames and Datasets
# MAGIC - Basic transformations and actions
# MAGIC - Working with different data formats
# MAGIC 
# MAGIC **Target Audience**: Beginners to Apache Spark
# MAGIC **Duration**: 30-45 minutes

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Understanding Spark Session
# MAGIC 
# MAGIC SparkSession is the entry point for all Spark functionality. In Databricks, it's automatically created as `spark`.

# COMMAND ----------

# Display Spark version and configuration
print(f"Spark Version: {spark.version}")
print(f"Application Name: {spark.sparkContext.appName}")
print(f"Master: {spark.sparkContext.master}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Creating DataFrames
# MAGIC 
# MAGIC DataFrames are the primary abstraction in Spark. Let's create some sample data to work with.

# COMMAND ----------

# Create a DataFrame from a list of tuples
data = [
    ("Alice", 25, "Engineer", 75000),
    ("Bob", 30, "Data Scientist", 85000),
    ("Charlie", 35, "Manager", 95000),
    ("Diana", 28, "Analyst", 65000),
    ("Eve", 32, "Engineer", 78000)
]

columns = ["name", "age", "job_title", "salary"]
df = spark.createDataFrame(data, columns)

# Display the DataFrame
display(df)

# COMMAND ----------

# Show DataFrame schema
df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Basic DataFrame Operations

# COMMAND ----------

# Basic DataFrame operations
print("Total rows:", df.count())
print("\nColumn names:", df.columns)
print("\nData types:")
df.dtypes

# COMMAND ----------

# Select specific columns
display(df.select("name", "salary"))

# COMMAND ----------

# Filter operations
display(df.filter(df.age > 30))

# COMMAND ----------

# Add a new column
from pyspark.sql.functions import col, when

df_with_category = df.withColumn(
    "salary_category",
    when(col("salary") > 80000, "High")
    .when(col("salary") > 70000, "Medium")
    .otherwise("Low")
)

display(df_with_category)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Aggregations and Grouping

# COMMAND ----------

# Group by operations
from pyspark.sql.functions import avg, max, min, count

# Average salary by job title
display(df.groupBy("job_title").agg(
    avg("salary").alias("avg_salary"),
    count("*").alias("count")
))

# COMMAND ----------

# Overall statistics
display(df.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Working with Built-in Datasets
# MAGIC 
# MAGIC Databricks provides sample datasets that are perfect for learning.

# COMMAND ----------

# Load a sample dataset
diamonds_df = spark.read.format("csv")\
    .option("header", "true")\
    .option("inferSchema", "true")\
    .load("/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv")

display(diamonds_df.limit(10))

# COMMAND ----------

# Explore the diamonds dataset
print("Number of records:", diamonds_df.count())
print("Number of columns:", len(diamonds_df.columns))
diamonds_df.printSchema()

# COMMAND ----------

# Basic analytics on diamonds dataset
display(diamonds_df.groupBy("cut").agg(
    avg("price").alias("avg_price"),
    count("*").alias("count")
).orderBy("avg_price", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Data Visualization
# MAGIC 
# MAGIC Databricks provides built-in visualization capabilities.

# COMMAND ----------

# Create a summary for visualization
cut_price_summary = diamonds_df.groupBy("cut").agg(
    avg("price").alias("avg_price"),
    count("*").alias("count")
).orderBy("avg_price", ascending=False)

display(cut_price_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Working with Different File Formats

# COMMAND ----------

# Save DataFrame as Parquet (efficient columnar format)
df.write.mode("overwrite").parquet("/tmp/employee_data.parquet")

# Read it back
parquet_df = spark.read.parquet("/tmp/employee_data.parquet")
display(parquet_df)

# COMMAND ----------

# Save as JSON
df.write.mode("overwrite").json("/tmp/employee_data.json")

# Read JSON
json_df = spark.read.json("/tmp/employee_data.json")
display(json_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. SQL Interface
# MAGIC 
# MAGIC Spark allows you to use SQL queries on DataFrames by creating temporary views.

# COMMAND ----------

# Create a temporary view
df.createOrReplaceTempView("employees")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   job_title,
# MAGIC   AVG(salary) as avg_salary,
# MAGIC   COUNT(*) as count,
# MAGIC   MAX(age) as max_age,
# MAGIC   MIN(age) as min_age
# MAGIC FROM employees
# MAGIC GROUP BY job_title
# MAGIC ORDER BY avg_salary DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Performance Tips for Beginners

# COMMAND ----------

# Show execution plan
df.filter(df.salary > 70000).explain()

# COMMAND ----------

# Cache frequently used DataFrames
cached_df = df.cache()
cached_df.count()  # This will cache the DataFrame

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Takeaways
# MAGIC 
# MAGIC 1. **SparkSession** is your entry point to Spark functionality
# MAGIC 2. **DataFrames** are the primary abstraction for structured data
# MAGIC 3. **Transformations** (like `select`, `filter`) are lazy - they don't execute immediately
# MAGIC 4. **Actions** (like `show`, `count`, `collect`) trigger execution
# MAGIC 5. **Spark SQL** provides a familiar interface for data analysis
# MAGIC 6. **Built-in datasets** are great for learning and experimentation
# MAGIC 7. **Caching** can improve performance for repeated operations
# MAGIC 
# MAGIC ## Next Steps
# MAGIC 
# MAGIC - Explore ETL pipelines in `02-etl/` notebooks
# MAGIC - Learn advanced analytics in `03-analytics/` notebooks
# MAGIC - Try machine learning examples in `04-ml/` notebooks

# COMMAND ----------
