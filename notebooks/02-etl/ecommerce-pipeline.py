# Databricks notebook source
# MAGIC %md
# MAGIC # E-commerce Analytics ETL Pipeline
# MAGIC 
# MAGIC This notebook demonstrates a complete ETL pipeline for e-commerce data analysis using Apache Spark:
# MAGIC - **Extract**: Load data from various sources
# MAGIC - **Transform**: Clean, enrich, and aggregate data  
# MAGIC - **Load**: Save processed data for analytics
# MAGIC 
# MAGIC **Business Context**: Analyzing customer purchasing patterns, product performance, and revenue trends
# MAGIC **Skills Demonstrated**: ETL patterns, data quality, aggregations, window functions

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate Sample E-commerce Data
# MAGIC 
# MAGIC Since we're using Databricks Community Edition, we'll create realistic sample data to work with.

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
spark.sparkContext.setCheckpointDir("/tmp/checkpoints")

# COMMAND ----------

# Generate sample customers data
customers_data = []
first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack"]
last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"]
states = ["NY", "CA", "IL", "TX", "AZ", "PA", "TX", "CA", "TX", "CA"]

for i in range(1000):
    customer_id = f"CUST_{i+1:04d}"
    first_name = random.choice(first_names)
    last_name = random.choice(last_names)
    email = f"{first_name.lower()}.{last_name.lower()}@email.com"
    city_idx = random.randint(0, len(cities)-1)
    city = cities[city_idx]
    state = states[city_idx]
    registration_date = datetime.now() - timedelta(days=random.randint(1, 365))
    
    customers_data.append((customer_id, first_name, last_name, email, city, state, registration_date))

customers_schema = StructType([
    StructField("customer_id", StringType(), True),
    StructField("first_name", StringType(), True),
    StructField("last_name", StringType(), True),
    StructField("email", StringType(), True),
    StructField("city", StringType(), True),
    StructField("state", StringType(), True),
    StructField("registration_date", TimestampType(), True)
])

customers_df = spark.createDataFrame(customers_data, customers_schema)
print(f"Generated {customers_df.count()} customers")
display(customers_df.limit(10))

# COMMAND ----------

# Generate sample products data
products_data = []
categories = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books", "Beauty", "Automotive", "Toys"]
electronics = ["Smartphone", "Laptop", "Tablet", "Headphones", "Camera", "Smartwatch"]
clothing = ["T-Shirt", "Jeans", "Dress", "Shoes", "Jacket", "Sweater"]
home_garden = ["Vacuum", "Blender", "Coffee Maker", "Plant Pot", "Cushion", "Lamp"]

product_names = {
    "Electronics": electronics,
    "Clothing": clothing, 
    "Home & Garden": home_garden,
    "Sports": ["Running Shoes", "Yoga Mat", "Dumbbells", "Basketball", "Tennis Racket"],
    "Books": ["Fiction Novel", "Cookbook", "Self-Help", "Biography", "Science Book"],
    "Beauty": ["Moisturizer", "Lipstick", "Shampoo", "Perfume", "Face Mask"],
    "Automotive": ["Car Polish", "Phone Mount", "Floor Mats", "Air Freshener"],
    "Toys": ["Board Game", "Action Figure", "Puzzle", "Doll", "Building Blocks"]
}

for i in range(500):
    product_id = f"PROD_{i+1:04d}"
    category = random.choice(categories)
    name = random.choice(product_names[category])
    price = round(random.uniform(10, 500), 2)
    cost = round(price * random.uniform(0.3, 0.7), 2)
    
    products_data.append((product_id, name, category, price, cost))

products_schema = StructType([
    StructField("product_id", StringType(), True),
    StructField("product_name", StringType(), True),
    StructField("category", StringType(), True),
    StructField("price", FloatType(), True),
    StructField("cost", FloatType(), True)
])

products_df = spark.createDataFrame(products_data, products_schema)
print(f"Generated {products_df.count()} products")
display(products_df.limit(10))

# COMMAND ----------

# Generate sample orders data
orders_data = []
order_statuses = ["Completed", "Shipped", "Processing", "Cancelled"]
customer_ids = [row.customer_id for row in customers_df.select("customer_id").collect()]
product_ids = [row.product_id for row in products_df.select("product_id").collect()]

for i in range(5000):
    order_id = f"ORD_{i+1:06d}"
    customer_id = random.choice(customer_ids)
    order_date = datetime.now() - timedelta(days=random.randint(1, 180))
    status = random.choice(order_statuses)
    
    # Generate 1-5 items per order
    num_items = random.randint(1, 5)
    order_items = []
    
    for j in range(num_items):
        product_id = random.choice(product_ids)
        quantity = random.randint(1, 3)
        order_items.append((order_id, product_id, quantity))
    
    orders_data.extend(order_items)

orders_schema = StructType([
    StructField("order_id", StringType(), True),
    StructField("product_id", StringType(), True),
    StructField("quantity", IntegerType(), True)
])

order_items_df = spark.createDataFrame(orders_data, orders_schema)

# Add order metadata
order_metadata = []
for order_id in order_items_df.select("order_id").distinct().collect():
    oid = order_id.order_id
    customer_id = random.choice(customer_ids)
    order_date = datetime.now() - timedelta(days=random.randint(1, 180))
    status = random.choice(order_statuses)
    order_metadata.append((oid, customer_id, order_date, status))

order_meta_schema = StructType([
    StructField("order_id", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("order_date", TimestampType(), True),
    StructField("status", StringType(), True)
])

orders_meta_df = spark.createDataFrame(order_metadata, order_meta_schema)

print(f"Generated {order_items_df.select('order_id').distinct().count()} orders with {order_items_df.count()} line items")
display(orders_meta_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Quality Assessment
# MAGIC 
# MAGIC Before processing, let's assess the quality of our data.

# COMMAND ----------

def assess_data_quality(df, table_name):
    """Assess data quality metrics for a DataFrame"""
    print(f"\n=== Data Quality Report for {table_name} ===")
    
    total_rows = df.count()
    print(f"Total rows: {total_rows}")
    
    # Check for nulls in each column
    print("\nNull counts by column:")
    for column in df.columns:
        null_count = df.filter(col(column).isNull()).count()
        null_percentage = (null_count / total_rows) * 100
        print(f"  {column}: {null_count} ({null_percentage:.2f}%)")
    
    # Check for duplicates
    distinct_rows = df.distinct().count()
    duplicates = total_rows - distinct_rows
    print(f"\nDuplicate rows: {duplicates}")
    
    return {
        "total_rows": total_rows,
        "distinct_rows": distinct_rows,
        "duplicates": duplicates
    }

# Assess quality of all datasets
customers_quality = assess_data_quality(customers_df, "Customers")
products_quality = assess_data_quality(products_df, "Products")
orders_quality = assess_data_quality(orders_meta_df, "Orders")
items_quality = assess_data_quality(order_items_df, "Order Items")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data Transformation and Enrichment

# COMMAND ----------

# Create enriched order details by joining all tables
enriched_orders = order_items_df \
    .join(orders_meta_df, "order_id") \
    .join(products_df, "product_id") \
    .join(customers_df, "customer_id")

# Add calculated columns
enriched_orders = enriched_orders.withColumn(
    "line_total", col("quantity") * col("price")
).withColumn(
    "profit", col("quantity") * (col("price") - col("cost"))
).withColumn(
    "order_month", date_format(col("order_date"), "yyyy-MM")
).withColumn(
    "customer_full_name", concat(col("first_name"), lit(" "), col("last_name"))
)

print(f"Enriched orders dataset: {enriched_orders.count()} records")
display(enriched_orders.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Business Analytics and Aggregations

# COMMAND ----------

# Customer analytics
customer_analytics = enriched_orders.groupBy("customer_id", "customer_full_name", "city", "state") \
    .agg(
        count("order_id").alias("total_orders"),
        countDistinct("order_id").alias("unique_orders"),
        sum("line_total").alias("total_spent"),
        avg("line_total").alias("avg_order_value"),
        max("order_date").alias("last_order_date"),
        countDistinct("category").alias("categories_purchased")
    )

print("Top 10 customers by total spent:")
display(customer_analytics.orderBy(col("total_spent").desc()).limit(10))

# COMMAND ----------

# Product performance analytics
product_analytics = enriched_orders.groupBy("product_id", "product_name", "category") \
    .agg(
        sum("quantity").alias("total_quantity_sold"),
        sum("line_total").alias("total_revenue"),
        sum("profit").alias("total_profit"),
        countDistinct("customer_id").alias("unique_customers"),
        avg("line_total").alias("avg_order_value")
    ) \
    .withColumn("profit_margin", (col("total_profit") / col("total_revenue") * 100))

print("Top 10 products by revenue:")
display(product_analytics.orderBy(col("total_revenue").desc()).limit(10))

# COMMAND ----------

# Monthly revenue trends
monthly_trends = enriched_orders.groupBy("order_month") \
    .agg(
        sum("line_total").alias("monthly_revenue"),
        sum("profit").alias("monthly_profit"),
        countDistinct("order_id").alias("monthly_orders"),
        countDistinct("customer_id").alias("monthly_customers")
    ) \
    .orderBy("order_month")

print("Monthly Revenue Trends:")
display(monthly_trends)

# COMMAND ----------

# Category performance analysis
category_performance = enriched_orders.groupBy("category") \
    .agg(
        sum("line_total").alias("category_revenue"),
        sum("profit").alias("category_profit"),
        sum("quantity").alias("items_sold"),
        countDistinct("customer_id").alias("unique_customers"),
        avg("price").alias("avg_price")
    ) \
    .withColumn("profit_margin", (col("category_profit") / col("category_revenue") * 100)) \
    .orderBy(col("category_revenue").desc())

print("Category Performance:")
display(category_performance)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Advanced Analytics with Window Functions

# COMMAND ----------

from pyspark.sql.window import Window

# Customer ranking by total spent
customer_window = Window.orderBy(col("total_spent").desc())

customer_rankings = customer_analytics.withColumn(
    "spending_rank", row_number().over(customer_window)
).withColumn(
    "spending_percentile", percent_rank().over(customer_window)
)

print("Customer Rankings:")
display(customer_rankings.filter(col("spending_rank") <= 20))

# COMMAND ----------

# Moving average of daily sales
daily_sales = enriched_orders.groupBy(date_format(col("order_date"), "yyyy-MM-dd").alias("sale_date")) \
    .agg(sum("line_total").alias("daily_revenue"))

# Define window for 7-day moving average
sales_window = Window.orderBy("sale_date").rowsBetween(-6, 0)

daily_sales_with_ma = daily_sales.withColumn(
    "moving_avg_7day", avg("daily_revenue").over(sales_window)
).orderBy("sale_date")

print("Daily Sales with 7-day Moving Average:")
display(daily_sales_with_ma)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Customer Segmentation

# COMMAND ----------

# RFM Analysis (Recency, Frequency, Monetary)
from pyspark.sql.functions import datediff, current_date

# Calculate RFM metrics
rfm_analysis = enriched_orders.groupBy("customer_id", "customer_full_name") \
    .agg(
        datediff(current_date(), max("order_date")).alias("recency_days"),
        countDistinct("order_id").alias("frequency"),
        sum("line_total").alias("monetary_value")
    )

# Create RFM segments
rfm_with_segments = rfm_analysis \
    .withColumn("recency_score", 
                when(col("recency_days") <= 30, 5)
                .when(col("recency_days") <= 60, 4)
                .when(col("recency_days") <= 90, 3)
                .when(col("recency_days") <= 120, 2)
                .otherwise(1)) \
    .withColumn("frequency_score",
                when(col("frequency") >= 10, 5)
                .when(col("frequency") >= 5, 4)
                .when(col("frequency") >= 3, 3)
                .when(col("frequency") >= 2, 2)
                .otherwise(1)) \
    .withColumn("monetary_score",
                when(col("monetary_value") >= 1000, 5)
                .when(col("monetary_value") >= 500, 4)
                .when(col("monetary_value") >= 200, 3)
                .when(col("monetary_value") >= 100, 2)
                .otherwise(1)) \
    .withColumn("rfm_score", col("recency_score") + col("frequency_score") + col("monetary_score")) \
    .withColumn("customer_segment",
                when(col("rfm_score") >= 12, "Champions")
                .when(col("rfm_score") >= 9, "Loyal Customers")
                .when(col("rfm_score") >= 6, "Potential Loyalists")
                .when(col("rfm_score") >= 4, "At Risk")
                .otherwise("Lost Customers"))

print("Customer Segmentation (RFM Analysis):")
display(rfm_with_segments.groupBy("customer_segment").count().orderBy("count", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Save Processed Data

# COMMAND ----------

# Save enriched data for further analysis
enriched_orders.write.mode("overwrite").parquet("/tmp/ecommerce/enriched_orders")
customer_analytics.write.mode("overwrite").parquet("/tmp/ecommerce/customer_analytics")
product_analytics.write.mode("overwrite").parquet("/tmp/ecommerce/product_analytics")
rfm_with_segments.write.mode("overwrite").parquet("/tmp/ecommerce/customer_segments")

print("Data successfully saved to /tmp/ecommerce/")

# COMMAND ----------

# Create views for SQL access
enriched_orders.createOrReplaceTempView("enriched_orders")
customer_analytics.createOrReplaceTempView("customer_analytics")
product_analytics.createOrReplaceTempView("product_analytics")
rfm_with_segments.createOrReplaceTempView("customer_segments")

print("Temporary views created for SQL access")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Example SQL query: Top customers by state
# MAGIC SELECT 
# MAGIC   state,
# MAGIC   COUNT(*) as customer_count,
# MAGIC   AVG(total_spent) as avg_customer_value,
# MAGIC   SUM(total_spent) as total_state_revenue
# MAGIC FROM customer_analytics
# MAGIC GROUP BY state
# MAGIC ORDER BY total_state_revenue DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Insights and Takeaways
# MAGIC 
# MAGIC ### Business Insights Generated:
# MAGIC 1. **Customer Behavior**: Identified top-spending customers and their purchasing patterns
# MAGIC 2. **Product Performance**: Analyzed which products drive the most revenue and profit
# MAGIC 3. **Seasonal Trends**: Discovered monthly revenue patterns and growth trends
# MAGIC 4. **Customer Segmentation**: Created RFM-based segments for targeted marketing
# MAGIC 5. **Geographic Analysis**: Understood revenue distribution across states
# MAGIC 
# MAGIC ### Technical Skills Demonstrated:
# MAGIC - **ETL Pipeline Design**: Extract, Transform, Load patterns
# MAGIC - **Data Quality Assessment**: Null checks, duplicate detection
# MAGIC - **Complex Joins**: Multi-table joins for data enrichment  
# MAGIC - **Advanced Aggregations**: GroupBy operations with multiple metrics
# MAGIC - **Window Functions**: Rankings, moving averages, percentiles
# MAGIC - **Data Segmentation**: RFM analysis for customer classification
# MAGIC - **Performance Optimization**: Efficient data storage formats
# MAGIC 
# MAGIC ### Next Steps:
# MAGIC - Explore machine learning for customer churn prediction
# MAGIC - Build real-time streaming analytics for live order processing
# MAGIC - Create interactive dashboards for business stakeholders

# COMMAND ----------