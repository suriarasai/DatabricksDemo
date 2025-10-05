# Databricks notebook source
# MAGIC %md
# MAGIC # Tutorial 4: ETL & Data Processing
# MAGIC
# MAGIC ## Overview
# MAGIC This notebook covers Extract, Transform, and Load (ETL) operations:
# MAGIC - Data transformation techniques
# MAGIC - Batch processing for large datasets
# MAGIC - Building data pipelines
# MAGIC - Data quality and validation
# MAGIC
# MAGIC **ETL Concepts:**
# MAGIC - **Extract**: Read data from various sources
# MAGIC - **Transform**: Clean, enrich, and reshape data
# MAGIC - **Load**: Write processed data to target destination

# COMMAND ----------

# Import libraries
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window
from datetime import datetime, timedelta
import re

print("Libraries imported successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Data Extraction
# MAGIC
# MAGIC ### Reading from Various Sources
# MAGIC
# MAGIC **Common Data Sources:**
# MAGIC - CSV files: `.read.csv()`
# MAGIC - JSON files: `.read.json()`
# MAGIC - Parquet files: `.read.parquet()`
# MAGIC - Delta tables: `.read.format("delta").load()`
# MAGIC - JDBC databases: `.read.jdbc()`

# COMMAND ----------

# Extract: Load raw data
print("=== EXTRACTING DATA ===")

# Read CSV files with options
customer_df = spark.read.csv(
    "/Volumes/workspace/sample/datasets/customer_data.csv",
    header=True,
    inferSchema=True,
    nullValue="NA",  # Treat "NA" as null
    mode="DROPMALFORMED"  # Drop malformed records
)

products_df = spark.read.csv(
    "/Volumes/workspace/sample/datasets/products.csv",
    header=True,
    inferSchema=True
)

sales_df = spark.read.csv(
    "/Volumes/workspace/sample/datasets/sales_data.csv",
    header=True,
    inferSchema=True
)

web_traffic_df = spark.read.csv(
    "/Volumes/workspace/sample/datasets/web_traffic.csv",
    header=True,
    inferSchema=True
)

print(f"Extracted: {customer_df.count()} customers")
print(f"Extracted: {products_df.count()} products")
print(f"Extracted: {sales_df.count()} sales records")
print(f"Extracted: {web_traffic_df.count()} traffic records")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Profiling After Extraction
# MAGIC
# MAGIC **Purpose:** Understand data quality issues before transformation

# COMMAND ----------

def profile_dataframe(df, name):
    """Profile a DataFrame to understand its characteristics"""
    print(f"\n=== PROFILING: {name} ===")
    print(f"Total Rows: {df.count():,}")
    print(f"Total Columns: {len(df.columns)}")
    
    # Check for nulls
    null_counts = df.select([
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) 
        for c in df.columns
    ]).collect()[0].asDict()
    
    print("\nNull Counts:")
    for col, count in null_counts.items():
        if count > 0:
            print(f"  {col}: {count} ({count/df.count()*100:.1f}%)")
    
    return null_counts

# Profile all datasets
customer_profile = profile_dataframe(customer_df, "CUSTOMERS")
products_profile = profile_dataframe(products_df, "PRODUCTS")
sales_profile = profile_dataframe(sales_df, "SALES")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Transformation
# MAGIC
# MAGIC ### String Transformations
# MAGIC
# MAGIC **Common Operations:**
# MAGIC - `.upper()`, `.lower()`: Change case
# MAGIC - `.trim()`: Remove whitespace
# MAGIC - `.regexp_replace()`: Pattern-based replacement
# MAGIC - `.concat()`: Combine strings
# MAGIC - `.substring()`: Extract substrings

# COMMAND ----------

# Transform: Clean and standardize customer data
print("=== TRANSFORMING CUSTOMER DATA ===")

customer_transformed = customer_df \
    .withColumn("first_name", F.trim(F.initcap(F.col("first_name")))) \
    .withColumn("last_name", F.trim(F.initcap(F.col("last_name")))) \
    .withColumn("email", F.lower(F.trim(F.col("email")))) \
    .withColumn("phone", F.regexp_replace(F.col("phone"), "[^0-9]", "")) \
    .withColumn("full_name", F.concat(F.col("first_name"), F.lit(" "), F.col("last_name"))) \
    .withColumn("state", F.upper(F.trim(F.col("state"))))

# Display sample
display(customer_transformed.select("first_name", "last_name", "full_name", "email", "phone").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Date and Time Transformations
# MAGIC
# MAGIC **Key Functions:**
# MAGIC - `to_date()`: Convert string to date
# MAGIC - `to_timestamp()`: Convert to timestamp
# MAGIC - `date_format()`: Format dates
# MAGIC - `datediff()`: Calculate date differences
# MAGIC - `add_months()`, `date_add()`: Date arithmetic

# COMMAND ----------

# Transform dates in customer data
customer_transformed = customer_transformed \
    .withColumn("registration_date", F.to_date(F.col("registration_date"))) \
    .withColumn("registration_year", F.year(F.col("registration_date"))) \
    .withColumn("registration_month", F.month(F.col("registration_date"))) \
    .withColumn("days_since_registration", 
                F.datediff(F.current_date(), F.col("registration_date"))) \
    .withColumn("customer_tenure_years", 
                F.round(F.col("days_since_registration") / 365, 1))

display(customer_transformed.select(
    "customer_id", "registration_date", "registration_year", 
    "days_since_registration", "customer_tenure_years"
).limit(10))

# COMMAND ----------

# Transform sales dates
sales_transformed = sales_df \
    .withColumn("date", F.to_date(F.col("date"))) \
    .withColumn("year", F.year(F.col("date"))) \
    .withColumn("month", F.month(F.col("date"))) \
    .withColumn("quarter", F.quarter(F.col("date"))) \
    .withColumn("day_of_week", F.dayofweek(F.col("date"))) \
    .withColumn("day_name", F.date_format(F.col("date"), "EEEE")) \
    .withColumn("is_weekend", F.when(F.col("day_of_week").isin([1, 7]), True).otherwise(False))

display(sales_transformed.select("date", "year", "month", "quarter", "day_name", "is_weekend").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numeric Transformations
# MAGIC
# MAGIC **Operations:**
# MAGIC - Mathematical operations: `+`, `-`, `*`, `/`
# MAGIC - Rounding: `round()`, `ceil()`, `floor()`
# MAGIC - Conditional logic: `when()`, `otherwise()`

# COMMAND ----------

# Transform product data with calculated fields
products_transformed = products_df \
    .withColumn("profit_margin", F.round((F.col("price") - F.col("cost")) / F.col("price") * 100, 2)) \
    .withColumn("price_tier", 
                F.when(F.col("price") < 50, "Budget")
                 .when(F.col("price") < 100, "Mid-Range")
                 .otherwise("Premium")) \
    .withColumn("rating_category",
                F.when(F.col("rating") >= 4.5, "Excellent")
                 .when(F.col("rating") >= 4.0, "Good")
                 .when(F.col("rating") >= 3.0, "Average")
                 .otherwise("Poor")) \
    .withColumn("review_volume",
                F.when(F.col("num_reviews") >= 1000, "High")
                 .when(F.col("num_reviews") >= 100, "Medium")
                 .otherwise("Low"))

display(products_transformed.select(
    "product_name", "price", "cost", "profit_margin", 
    "price_tier", "rating", "rating_category"
).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Type Conversions
# MAGIC
# MAGIC **Cast Operations:**
# MAGIC - `.cast("string")`: Convert to string
# MAGIC - `.cast("integer")`: Convert to integer
# MAGIC - `.cast("double")`: Convert to double
# MAGIC - `.cast("date")`: Convert to date

# COMMAND ----------

# Ensure correct data types
customer_typed = customer_transformed \
    .withColumn("customer_id", F.col("customer_id").cast("string")) \
    .withColumn("age", F.col("age").cast("integer")) \
    .withColumn("annual_income", F.col("annual_income").cast("double")) \
    .withColumn("email_subscribed", F.col("email_subscribed").cast("boolean"))

print("Data types corrected")
customer_typed.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Handling Missing Values
# MAGIC
# MAGIC **Strategies:**
# MAGIC 1. Drop rows with nulls: `.na.drop()`
# MAGIC 2. Fill with constants: `.na.fill()`
# MAGIC 3. Fill with statistics: Use mean, median, mode
# MAGIC 4. Forward/backward fill: Use window functions

# COMMAND ----------

# Handle missing values with different strategies

# Strategy 1: Fill with defaults
customer_filled = customer_typed.na.fill({
    "phone": "0000000000",
    "email_subscribed": False,
    "annual_income": 0
})

# Strategy 2: Fill with column mean
avg_age = customer_typed.select(F.avg("age")).first()[0]
customer_filled = customer_filled.na.fill({"age": int(avg_age)})

# Strategy 3: Fill categorical with mode
mode_segment = customer_typed.groupBy("segment").count().orderBy(F.desc("count")).first()["segment"]
customer_filled = customer_filled.na.fill({"segment": mode_segment})

print(f"Missing values handled. Records: {customer_filled.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Deduplication
# MAGIC
# MAGIC **Methods:**
# MAGIC - Remove exact duplicates: `.dropDuplicates()`
# MAGIC - Remove duplicates by key: `.dropDuplicates([columns])`
# MAGIC - Keep first/last occurrence using window functions

# COMMAND ----------

# Remove duplicates
print("=== DEDUPLICATION ===")
print(f"Original records: {customer_filled.count()}")

# Remove exact duplicates
customer_deduped = customer_filled.dropDuplicates()
print(f"After removing exact duplicates: {customer_deduped.count()}")

# Remove duplicates by customer_id, keeping the most recent
window_spec = Window.partitionBy("customer_id").orderBy(F.desc("registration_date"))
customer_unique = customer_filled \
    .withColumn("row_num", F.row_number().over(window_spec)) \
    .filter(F.col("row_num") == 1) \
    .drop("row_num")

print(f"After keeping latest record per customer: {customer_unique.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Enrichment
# MAGIC
# MAGIC **Adding Derived Features:**
# MAGIC - Create new columns based on existing data
# MAGIC - Aggregate features from related data
# MAGIC - Add external reference data

# COMMAND ----------

# Enrich customer data with calculated features
customer_enriched = customer_unique \
    .withColumn("income_bracket",
                F.when(F.col("annual_income") < 30000, "Low")
                 .when(F.col("annual_income") < 60000, "Lower-Middle")
                 .when(F.col("annual_income") < 100000, "Middle")
                 .when(F.col("annual_income") < 150000, "Upper-Middle")
                 .otherwise("High")) \
    .withColumn("age_group",
                F.when(F.col("age") < 25, "18-24")
                 .when(F.col("age") < 35, "25-34")
                 .when(F.col("age") < 45, "35-44")
                 .when(F.col("age") < 55, "45-54")
                 .when(F.col("age") < 65, "55-64")
                 .otherwise("65+")) \
    .withColumn("customer_value_score",
                F.when((F.col("segment") == "VIP") & (F.col("email_subscribed") == True), 100)
                 .when((F.col("segment") == "Premium") & (F.col("email_subscribed") == True), 80)
                 .when(F.col("segment") == "VIP", 70)
                 .when((F.col("segment") == "Premium") & (F.col("email_subscribed") == False), 60)
                 .when((F.col("segment") == "Standard") & (F.col("email_subscribed") == True), 50)
                 .otherwise(30))

display(customer_enriched.select(
    "customer_id", "age", "age_group", "annual_income", 
    "income_bracket", "segment", "customer_value_score"
).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Batch Processing
# MAGIC
# MAGIC ### Processing Large Datasets Efficiently
# MAGIC
# MAGIC **Techniques:**
# MAGIC - Partitioning data
# MAGIC - Using appropriate file formats (Parquet, Delta)
# MAGIC - Caching intermediate results
# MAGIC - Broadcast joins for small lookup tables

# COMMAND ----------

# Repartition for better performance
print("=== BATCH PROCESSING ===")

# Check current partitions
# print(f"Current partitions: {customer_enriched.rdd.getNumPartitions()}")

# Repartition based on data size (rule of thumb: 128MB per partition)
# optimal_partitions = max(1, customer_enriched.count() // 10000)
# customer_partitioned = customer_enriched.repartition(optimal_partitions, "state")

# print(f"Optimized partitions: {customer_partitioned.rdd.getNumPartitions()}")

# COMMAND ----------

# Batch processing example: Process data in chunks
def process_customer_batch(batch_df):
    """Process a batch of customer data"""
    return batch_df \
        .withColumn("processed_date", F.current_timestamp()) \
        .withColumn("batch_id", F.lit(datetime.now().strftime("%Y%m%d%H%M%S")))

# Process all data
customer_processed = process_customer_batch(customer_partitioned)
print(f"Batch processing complete: {customer_processed.count()} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Aggregations for Batch Processing
# MAGIC
# MAGIC **Efficient Aggregations:**
# MAGIC - Use built-in aggregate functions
# MAGIC - Avoid UDFs when possible
# MAGIC - Leverage Spark's optimization

# COMMAND ----------

# Aggregate customer metrics by segment and state
customer_aggregated = customer_processed.groupBy("state", "segment").agg(
    F.count("customer_id").alias("customer_count"),
    F.avg("age").alias("avg_age"),
    F.avg("annual_income").alias("avg_income"),
    F.min("annual_income").alias("min_income"),
    F.max("annual_income").alias("max_income"),
    F.stddev("annual_income").alias("stddev_income"),
    F.sum(F.when(F.col("email_subscribed") == True, 1).otherwise(0)).alias("subscribed_count"),
    F.avg("customer_value_score").alias("avg_value_score")
)

display(customer_aggregated.orderBy(F.desc("customer_count")).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Building Data Pipelines
# MAGIC
# MAGIC ### Pipeline Structure
# MAGIC
# MAGIC **Components:**
# MAGIC 1. Data ingestion
# MAGIC 2. Data validation
# MAGIC 3. Data transformation
# MAGIC 4. Data quality checks
# MAGIC 5. Data loading

# COMMAND ----------

# MAGIC %md
# MAGIC ### Complete ETL Pipeline Example

# COMMAND ----------

def etl_pipeline_customer_data(source_path, target_path):
    """
    Complete ETL pipeline for customer data
    
    Steps:
    1. Extract: Read raw data
    2. Validate: Check data quality
    3. Transform: Clean and enrich
    4. Quality Check: Validate output
    5. Load: Write to target
    """
    
    print("=== STARTING ETL PIPELINE ===")
    
    # STEP 1: EXTRACT
    print("\n1. Extracting data...")
    raw_df = spark.read.csv(source_path, header=True, inferSchema=True)
    print(f"   Extracted: {raw_df.count()} records")
    
    # STEP 2: VALIDATE INPUT
    print("\n2. Validating input data...")
    null_counts = raw_df.select([
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) 
        for c in raw_df.columns
    ]).collect()[0].asDict()
    
    total_nulls = sum(null_counts.values())
    print(f"   Total null values: {total_nulls}")
    
    # STEP 3: TRANSFORM
    print("\n3. Transforming data...")
    
    # Clean strings
    transformed_df = raw_df \
        .withColumn("first_name", F.trim(F.initcap(F.col("first_name")))) \
        .withColumn("last_name", F.trim(F.initcap(F.col("last_name")))) \
        .withColumn("email", F.lower(F.trim(F.col("email")))) \
        .withColumn("full_name", F.concat(F.col("first_name"), F.lit(" "), F.col("last_name")))
    
    # Convert dates
    transformed_df = transformed_df \
        .withColumn("registration_date", F.to_date(F.col("registration_date"))) \
        .withColumn("customer_tenure_years", 
                    F.round(F.datediff(F.current_date(), F.col("registration_date")) / 365, 1))
    
    # Handle nulls
    avg_age = transformed_df.select(F.avg("age")).first()[0]
    transformed_df = transformed_df.na.fill({
        "age": int(avg_age),
        "email_subscribed": False
    })
    
    # Remove duplicates
    transformed_df = transformed_df.dropDuplicates(["customer_id"])
    
    # Add enrichment
    transformed_df = transformed_df \
        .withColumn("age_group",
                    F.when(F.col("age") < 25, "18-24")
                     .when(F.col("age") < 35, "25-34")
                     .when(F.col("age") < 45, "35-44")
                     .when(F.col("age") < 55, "45-54")
                     .when(F.col("age") < 65, "55-64")
                     .otherwise("65+")) \
        .withColumn("processed_timestamp", F.current_timestamp()) \
        .withColumn("pipeline_version", F.lit("1.0"))
    
    print(f"   Transformed: {transformed_df.count()} records")
    
    # STEP 4: QUALITY CHECKS
    print("\n4. Running quality checks...")
    
    # Check 1: No nulls in critical columns
    critical_columns = ["customer_id", "email", "registration_date"]
    null_check = transformed_df.select([
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) 
        for c in critical_columns
    ]).collect()[0].asDict()
    
    if sum(null_check.values()) > 0:
        print("   WARNING: Null values found in critical columns!")
        return None
    else:
        print("   ✓ No nulls in critical columns")
    
    # Check 2: Valid email format
    email_pattern = "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
    invalid_emails = transformed_df.filter(~F.col("email").rlike(email_pattern)).count()
    print(f"   ✓ Valid emails: {transformed_df.count() - invalid_emails}/{transformed_df.count()}")
    
    # Check 3: Age range
    age_check = transformed_df.filter((F.col("age") < 18) | (F.col("age") > 100)).count()
    print(f"   ✓ Valid age range: {transformed_df.count() - age_check}/{transformed_df.count()}")
    
    # STEP 5: LOAD
    print("\n5. Loading data...")
    transformed_df.write.mode("overwrite").format("delta").save(target_path)
    print(f"   ✓ Data loaded to: {target_path}")
    
    print("\n=== ETL PIPELINE COMPLETED SUCCESSFULLY ===")
    return transformed_df

# Execute pipeline
result_df = etl_pipeline_customer_data(
    "/Volumes/workspace/sample/datasets/customer_data.csv",
    "/Volumes/workspace/sample/datasets/customers_clean"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pipeline for Sales Data

# COMMAND ----------

def etl_pipeline_sales_data(source_path, target_path):
    """ETL pipeline for sales data"""
    
    print("=== SALES DATA ETL PIPELINE ===")
    
    # Extract
    print("\n1. Extracting sales data...")
    sales_raw = spark.read.csv(source_path, header=True, inferSchema=True)
    print(f"   Extracted: {sales_raw.count()} transactions")
    
    # Transform
    print("\n2. Transforming sales data...")
    sales_transformed = sales_raw \
        .withColumn("date", F.to_date(F.col("date"))) \
        .withColumn("year", F.year(F.col("date"))) \
        .withColumn("month", F.month(F.col("date"))) \
        .withColumn("quarter", F.quarter(F.col("date"))) \
        .withColumn("revenue_category",
                    F.when(F.col("total_sales") < 100, "Small")
                     .when(F.col("total_sales") < 500, "Medium")
                     .when(F.col("total_sales") < 1000, "Large")
                     .otherwise("Enterprise")) \
        .withColumn("satisfaction_level",
                    F.when(F.col("customer_satisfaction") >= 4.5, "Excellent")
                     .when(F.col("customer_satisfaction") >= 4.0, "Good")
                     .when(F.col("customer_satisfaction") >= 3.0, "Average")
                     .otherwise("Poor")) \
        .withColumn("processed_timestamp", F.current_timestamp())
    
    # Add calculated metrics
    sales_transformed = sales_transformed \
        .withColumn("avg_unit_price", F.round(F.col("total_sales") / F.col("quantity"), 2))
    
    # Quality checks
    print("\n3. Quality checks...")
    invalid_dates = sales_transformed.filter(F.col("date").isNull()).count()
    negative_sales = sales_transformed.filter(F.col("total_sales") < 0).count()
    
    print(f"   ✓ Valid dates: {sales_transformed.count() - invalid_dates}/{sales_transformed.count()}")
    print(f"   ✓ Non-negative sales: {sales_transformed.count() - negative_sales}/{sales_transformed.count()}")
    
    # Load
    print("\n4. Loading transformed sales data...")
    sales_transformed.write.mode("overwrite").partitionBy("year", "month").format("delta").save(target_path)
    print(f"   ✓ Data loaded and partitioned by year/month")
    
    print("\n=== SALES ETL PIPELINE COMPLETED ===")
    return sales_transformed

# Execute sales pipeline
sales_clean = etl_pipeline_sales_data(
    "/Volumes/workspace/sample/datasets/sales_data.csv",
    "/Volumes/workspace/sample/datasets/sales_clean"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Incremental Data Processing
# MAGIC
# MAGIC **Purpose:** Process only new or changed data
# MAGIC
# MAGIC **Approach:**
# MAGIC - Track last processed timestamp
# MAGIC - Filter for new records
# MAGIC - Merge with existing data

# COMMAND ----------

# Simulate incremental processing
def incremental_etl(source_df, last_processed_date):
    """
    Process only records after the last processed date
    """
    print(f"Processing records after: {last_processed_date}")
    
    # Filter for new records
    new_records = source_df.filter(F.col("date") > last_processed_date)
    
    print(f"New records to process: {new_records.count()}")
    
    # Transform new records
    transformed = new_records \
        .withColumn("processed_timestamp", F.current_timestamp()) \
        .withColumn("is_incremental", F.lit(True))
    
    return transformed

# Example: Process sales from last 30 days
cutoff_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
incremental_sales = incremental_etl(sales_clean, cutoff_date)

display(incremental_sales.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Data Quality Framework
# MAGIC
# MAGIC ### Implementing Data Quality Checks

# COMMAND ----------

def data_quality_report(df, name):
    """
    Generate comprehensive data quality report
    """
    print(f"\n{'='*60}")
    print(f"DATA QUALITY REPORT: {name}")
    print(f"{'='*60}")
    
    # Basic stats
    total_rows = df.count()
    total_cols = len(df.columns)
    print(f"\nDataset Size: {total_rows:,} rows × {total_cols} columns")
    
    # Completeness check
    print("\n--- COMPLETENESS ---")
    null_counts = df.select([
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) 
        for c in df.columns
    ]).collect()[0].asDict()
    
    for col, null_count in null_counts.items():
        completeness = (1 - null_count/total_rows) * 100
        status = "✓" if completeness == 100 else "⚠"
        print(f"{status} {col}: {completeness:.1f}% complete")
    
    # Uniqueness check
    print("\n--- UNIQUENESS ---")
    for col in df.columns:
        unique_count = df.select(col).distinct().count()
        uniqueness = (unique_count/total_rows) * 100
        print(f"  {col}: {uniqueness:.1f}% unique ({unique_count:,} distinct values)")
    
    # Duplicate check
    print("\n--- DUPLICATES ---")
    duplicate_count = total_rows - df.dropDuplicates().count()
    print(f"  Exact duplicate rows: {duplicate_count}")
    
    print(f"\n{'='*60}\n")

# Generate quality reports
data_quality_report(customer_enriched, "CUSTOMERS")
data_quality_report(sales_clean, "SALES")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Takeaways
# MAGIC
# MAGIC **ETL Best Practices:**
# MAGIC - Always profile data before transformation
# MAGIC - Document transformation logic
# MAGIC - Handle errors gracefully
# MAGIC - Implement data quality checks
# MAGIC - Use appropriate file formats (Delta, Parquet)
# MAGIC
# MAGIC **Data Transformation:**
# MAGIC - Clean strings (trim, case conversion)
# MAGIC - Standardize dates and times
# MAGIC - Handle missing values appropriately
# MAGIC - Remove duplicates strategically
# MAGIC - Enrich data with derived features
# MAGIC
# MAGIC **Batch Processing:**
# MAGIC - Partition data appropriately
# MAGIC - Use caching for iterative operations
# MAGIC - Leverage Spark's built-in optimizations
# MAGIC - Monitor performance metrics
# MAGIC
# MAGIC **Data Pipelines:**
# MAGIC - Follow Extract → Transform → Load pattern
# MAGIC - Implement quality checks at each stage
# MAGIC - Design for incremental processing
# MAGIC - Create reusable pipeline components
# MAGIC
# MAGIC **Next Steps:** Move to Notebook 5 for Machine Learning!
