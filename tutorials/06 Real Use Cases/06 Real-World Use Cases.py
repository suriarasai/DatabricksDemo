# Databricks notebook source
# MAGIC %md
# MAGIC # Tutorial 6: Real-World Use Cases
# MAGIC
# MAGIC ## Overview
# MAGIC This notebook demonstrates practical business applications:
# MAGIC - **Sales Analytics**: Performance analysis and trend identification
# MAGIC - **Customer Segmentation**: RFM analysis and behavioral grouping
# MAGIC - **Time Series Analysis**: Forecasting and trend decomposition
# MAGIC
# MAGIC **Business Value:**
# MAGIC - Actionable insights from data
# MAGIC - Data-driven decision making
# MAGIC - Revenue optimization
# MAGIC - Customer understanding

# COMMAND ----------

# Import libraries
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ML libraries
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

print("Libraries imported successfully!")

# COMMAND ----------

# Load all datasets
customer_df = spark.read.csv("/Volumes/workspace/sample/datasets/customer_data.csv", header=True, inferSchema=True)
products_df = spark.read.csv("/Volumes/workspace/sample/datasets/products.csv", header=True, inferSchema=True)
sales_df = spark.read.csv("/Volumes/workspace/sample/datasets/sales_data.csv", header=True, inferSchema=True)
web_traffic_df = spark.read.csv("/Volumes/workspace/sample/datasets/web_traffic.csv", header=True, inferSchema=True)

print("All datasets loaded!")
print(f"Customers: {customer_df.count()}")
print(f"Products: {products_df.count()}")
print(f"Sales: {sales_df.count()}")
print(f"Web Traffic: {web_traffic_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Use Case 1: Sales Analytics
# MAGIC
# MAGIC ## Objective: Analyze sales performance and identify trends
# MAGIC
# MAGIC **Business Questions:**
# MAGIC 1. What are our top-performing products and regions?
# MAGIC 2. How do sales vary over time?
# MAGIC 3. Which sales representatives are most effective?
# MAGIC 4. What factors drive customer satisfaction?
# MAGIC 5. Where should we focus our resources?

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Sales Performance Overview

# COMMAND ----------

# Prepare sales data with date transformations
sales_analysis = sales_df \
    .withColumn("date", F.to_date(F.col("date"))) \
    .withColumn("year", F.year(F.col("date"))) \
    .withColumn("month", F.month(F.col("date"))) \
    .withColumn("quarter", F.quarter(F.col("date"))) \
    .withColumn("day_of_week", F.dayofweek(F.col("date"))) \
    .withColumn("month_name", F.date_format(F.col("date"), "MMMM"))

# Overall sales metrics
print("=== OVERALL SALES PERFORMANCE ===\n")

total_metrics = sales_analysis.agg(
    F.sum("total_sales").alias("total_revenue"),
    F.count("transaction_id").alias("total_transactions"),
    F.avg("total_sales").alias("avg_transaction_value"),
    F.sum("quantity").alias("total_units_sold"),
    F.avg("customer_satisfaction").alias("avg_satisfaction"),
    F.countDistinct("product").alias("unique_products_sold"),
    F.countDistinct("sales_rep").alias("total_sales_reps")
).collect()[0]

print(f"Total Revenue: ${total_metrics['total_revenue']:,.2f}")
print(f"Total Transactions: {total_metrics['total_transactions']:,}")
print(f"Average Transaction Value: ${total_metrics['avg_transaction_value']:,.2f}")
print(f"Total Units Sold: {total_metrics['total_units_sold']:,}")
print(f"Average Customer Satisfaction: {total_metrics['avg_satisfaction']:.2f}/5.0")
print(f"Unique Products Sold: {total_metrics['unique_products_sold']}")
print(f"Active Sales Representatives: {total_metrics['total_sales_reps']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 Top Performers Analysis

# COMMAND ----------

# Top 10 products by revenue
top_products = sales_analysis.groupBy("product").agg(
    F.sum("total_sales").alias("revenue"),
    F.sum("quantity").alias("units_sold"),
    F.count("transaction_id").alias("transactions"),
    F.avg("customer_satisfaction").alias("avg_satisfaction")
).orderBy(F.desc("revenue")).limit(10)

print("\n=== TOP 10 PRODUCTS BY REVENUE ===")
display(top_products)

# COMMAND ----------

# Visualize top products
top_products_pd = top_products.toPandas()

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Revenue by Product', 'Customer Satisfaction'),
    specs=[[{"type": "bar"}, {"type": "bar"}]]
)

# Revenue chart
fig.add_trace(
    go.Bar(x=top_products_pd['product'], y=top_products_pd['revenue'],
           name='Revenue', marker_color='steelblue'),
    row=1, col=1
)

# Satisfaction chart
fig.add_trace(
    go.Bar(x=top_products_pd['product'], y=top_products_pd['avg_satisfaction'],
           name='Satisfaction', marker_color='lightcoral'),
    row=1, col=2
)

fig.update_xaxes(tickangle=45)
fig.update_layout(height=500, showlegend=False, title_text="Top 10 Products Performance")
fig.show()

# COMMAND ----------

# Regional performance analysis
regional_performance = sales_analysis.groupBy("region").agg(
    F.sum("total_sales").alias("revenue"),
    F.count("transaction_id").alias("transactions"),
    F.avg("total_sales").alias("avg_transaction"),
    F.avg("customer_satisfaction").alias("avg_satisfaction"),
    F.sum("quantity").alias("units_sold")
).orderBy(F.desc("revenue"))

print("\n=== REGIONAL PERFORMANCE ===")
display(regional_performance)

# COMMAND ----------

# Regional performance visualization
regional_pd = regional_performance.toPandas()

fig = px.bar(regional_pd, x='region', y='revenue',
             title='Revenue by Region',
             labels={'revenue': 'Total Revenue ($)', 'region': 'Region'},
             color='avg_satisfaction',
             color_continuous_scale='RdYlGn',
             text='revenue')

fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
fig.update_layout(height=500)
fig.show()

# COMMAND ----------

# Sales representative performance
rep_performance = sales_analysis.groupBy("sales_rep").agg(
    F.sum("total_sales").alias("revenue"),
    F.count("transaction_id").alias("transactions"),
    F.avg("customer_satisfaction").alias("avg_satisfaction"),
    F.sum("quantity").alias("units_sold")
).orderBy(F.desc("revenue"))

print("\n=== TOP 15 SALES REPRESENTATIVES ===")
display(rep_performance.limit(15))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.3 Time-Based Sales Analysis

# COMMAND ----------

# Daily sales trend
daily_sales = sales_analysis.groupBy("date").agg(
    F.sum("total_sales").alias("daily_revenue"),
    F.count("transaction_id").alias("transactions"),
    F.avg("customer_satisfaction").alias("avg_satisfaction")
).orderBy("date")

display(daily_sales)

# COMMAND ----------

# Monthly sales trend
monthly_sales = sales_analysis.groupBy("year", "month", "month_name").agg(
    F.sum("total_sales").alias("revenue"),
    F.count("transaction_id").alias("transactions"),
    F.avg("total_sales").alias("avg_transaction"),
    F.avg("customer_satisfaction").alias("avg_satisfaction")
).orderBy("year", "month")

print("\n=== MONTHLY SALES TREND ===")
display(monthly_sales)

# COMMAND ----------

# Visualize monthly trend
monthly_pd = monthly_sales.toPandas()
monthly_pd['month_year'] = monthly_pd['year'].astype(str) + '-' + monthly_pd['month'].astype(str).str.zfill(2)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    subplot_titles=('Monthly Revenue', 'Monthly Transactions'),
                    vertical_spacing=0.1)

# Revenue trend
fig.add_trace(
    go.Scatter(x=monthly_pd['month_year'], y=monthly_pd['revenue'],
               mode='lines+markers', name='Revenue',
               line=dict(color='blue', width=2)),
    row=1, col=1
)

# Transaction trend
fig.add_trace(
    go.Scatter(x=monthly_pd['month_year'], y=monthly_pd['transactions'],
               mode='lines+markers', name='Transactions',
               line=dict(color='green', width=2)),
    row=2, col=1
)

fig.update_xaxes(tickangle=45)
fig.update_layout(height=600, title_text="Sales Trends Over Time")
fig.show()

# COMMAND ----------

# Day of week analysis
dow_sales = sales_analysis.groupBy("day_of_week").agg(
    F.sum("total_sales").alias("revenue"),
    F.count("transaction_id").alias("transactions"),
    F.avg("customer_satisfaction").alias("avg_satisfaction")
).orderBy("day_of_week")

# Add day names
dow_sales = dow_sales.withColumn("day_name",
    F.when(F.col("day_of_week") == 1, "Sunday")
     .when(F.col("day_of_week") == 2, "Monday")
     .when(F.col("day_of_week") == 3, "Tuesday")
     .when(F.col("day_of_week") == 4, "Wednesday")
     .when(F.col("day_of_week") == 5, "Thursday")
     .when(F.col("day_of_week") == 6, "Friday")
     .otherwise("Saturday")
)

print("\n=== DAY OF WEEK ANALYSIS ===")
display(dow_sales.select("day_name", "revenue", "transactions", "avg_satisfaction"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.4 Advanced Sales Insights

# COMMAND ----------

# Moving averages for trend analysis
window_spec = Window.orderBy("date").rowsBetween(-6, 0)

daily_with_ma = daily_sales.withColumn(
    "revenue_ma_7day",
    F.avg("daily_revenue").over(window_spec)
).withColumn(
    "transactions_ma_7day",
    F.avg("transactions").over(window_spec)
)

# Calculate growth rates
window_lag = Window.orderBy("date")

daily_with_growth = daily_with_ma.withColumn(
    "prev_day_revenue",
    F.lag("daily_revenue", 1).over(window_lag)
).withColumn(
    "revenue_growth_pct",
    F.round((F.col("daily_revenue") - F.col("prev_day_revenue")) / F.col("prev_day_revenue") * 100, 2)
)

print("\n=== DAILY SALES WITH MOVING AVERAGE ===")
display(daily_with_growth.orderBy(F.desc("date")).limit(30))

# COMMAND ----------

# Sales performance matrix
performance_matrix = sales_analysis.groupBy("region", "quarter").agg(
    F.sum("total_sales").alias("revenue"),
    F.avg("customer_satisfaction").alias("avg_satisfaction")
).orderBy("region", "quarter")

# Pivot for heatmap
performance_pivot = performance_matrix.groupBy("region").pivot("quarter").agg(
    F.first("revenue")
)

print("\n=== QUARTERLY REVENUE BY REGION ===")
display(performance_pivot)

# COMMAND ----------

# MAGIC %md
# MAGIC # Use Case 2: Customer Segmentation
# MAGIC
# MAGIC ## Objective: Group customers based on behavior and value
# MAGIC
# MAGIC **Approach: RFM Analysis**
# MAGIC - **Recency**: How recently did the customer register?
# MAGIC - **Frequency**: How engaged is the customer? (email subscription)
# MAGIC - **Monetary**: What is the customer's income level?

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 RFM Analysis

# COMMAND ----------

# Prepare customer data for RFM
print("=== CUSTOMER SEGMENTATION: RFM ANALYSIS ===\n")

# Calculate RFM metrics
customer_rfm = customer_df \
    .withColumn("registration_date", F.to_date(F.col("registration_date"))) \
    .withColumn("recency_days", F.datediff(F.current_date(), F.col("registration_date"))) \
    .withColumn("frequency_score", F.when(F.col("email_subscribed") == True, 2).otherwise(1)) \
    .withColumn("monetary_value", F.col("annual_income"))

# Calculate RFM scores (1-5 scale)
# Recency: Lower is better (more recent)
recency_quantiles = customer_rfm.approxQuantile("recency_days", [0.2, 0.4, 0.6, 0.8], 0.01)

customer_rfm = customer_rfm.withColumn("R_score",
    F.when(F.col("recency_days") <= recency_quantiles[0], 5)
     .when(F.col("recency_days") <= recency_quantiles[1], 4)
     .when(F.col("recency_days") <= recency_quantiles[2], 3)
     .when(F.col("recency_days") <= recency_quantiles[3], 2)
     .otherwise(1)
)

# Frequency: Email subscribed gets higher score
customer_rfm = customer_rfm.withColumn("F_score",
    F.when(F.col("email_subscribed") == True, 5).otherwise(2)
)

# Monetary: Higher income = higher score
monetary_quantiles = customer_rfm.approxQuantile("annual_income", [0.2, 0.4, 0.6, 0.8], 0.01)

customer_rfm = customer_rfm.withColumn("M_score",
    F.when(F.col("annual_income") >= monetary_quantiles[3], 5)
     .when(F.col("annual_income") >= monetary_quantiles[2], 4)
     .when(F.col("annual_income") >= monetary_quantiles[1], 3)
     .when(F.col("annual_income") >= monetary_quantiles[0], 2)
     .otherwise(1)
)

# Calculate total RFM score
customer_rfm = customer_rfm.withColumn("RFM_score", 
    F.col("R_score") + F.col("F_score") + F.col("M_score")
)

display(customer_rfm.select("customer_id", "first_name", "last_name", 
                            "recency_days", "R_score", "F_score", "M_score", 
                            "RFM_score", "annual_income").limit(20))

# COMMAND ----------

# Create customer segments based on RFM
customer_rfm = customer_rfm.withColumn("customer_segment",
    F.when(F.col("RFM_score") >= 13, "Champions")
     .when(F.col("RFM_score") >= 11, "Loyal Customers")
     .when(F.col("RFM_score") >= 9, "Potential Loyalists")
     .when(F.col("RFM_score") >= 7, "At Risk")
     .when(F.col("RFM_score") >= 5, "Need Attention")
     .otherwise("Lost")
)

# Segment distribution
segment_distribution = customer_rfm.groupBy("customer_segment").agg(
    F.count("customer_id").alias("customer_count"),
    F.avg("annual_income").alias("avg_income"),
    F.avg("age").alias("avg_age"),
    F.sum(F.when(F.col("email_subscribed") == True, 1).otherwise(0)).alias("subscribed_count")
).orderBy(F.desc("customer_count"))

print("\n=== CUSTOMER SEGMENTATION RESULTS ===")
display(segment_distribution)

# COMMAND ----------

# Visualize segment distribution
segment_pd = segment_distribution.toPandas()

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "pie"}, {"type": "bar"}]],
    subplot_titles=('Customer Distribution by Segment', 'Average Income by Segment')
)

# Pie chart
fig.add_trace(
    go.Pie(labels=segment_pd['customer_segment'], 
           values=segment_pd['customer_count'],
           hole=0.3),
    row=1, col=1
)

# Bar chart
fig.add_trace(
    go.Bar(x=segment_pd['customer_segment'], 
           y=segment_pd['avg_income'],
           marker_color='lightblue'),
    row=1, col=2
)

fig.update_layout(height=500, title_text="Customer Segmentation Analysis")
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 Demographic Segmentation

# COMMAND ----------

# Age group analysis
customer_demo = customer_rfm.withColumn("age_group",
    F.when(F.col("age") < 25, "18-24")
     .when(F.col("age") < 35, "25-34")
     .when(F.col("age") < 45, "35-44")
     .when(F.col("age") < 55, "45-54")
     .when(F.col("age") < 65, "55-64")
     .otherwise("65+")
).withColumn("income_bracket",
    F.when(F.col("annual_income") < 30000, "< $30K")
     .when(F.col("annual_income") < 60000, "$30K-$60K")
     .when(F.col("annual_income") < 100000, "$60K-$100K")
     .otherwise("$100K+")
)

# Age group distribution
age_analysis = customer_demo.groupBy("age_group").agg(
    F.count("customer_id").alias("count"),
    F.avg("annual_income").alias("avg_income")
).orderBy("age_group")

print("\n=== AGE GROUP ANALYSIS ===")
display(age_analysis)

# COMMAND ----------

# Income bracket distribution
income_analysis = customer_demo.groupBy("income_bracket").agg(
    F.count("customer_id").alias("count"),
    F.avg("age").alias("avg_age")
).orderBy("income_bracket")

print("\n=== INCOME BRACKET ANALYSIS ===")
display(income_analysis)

# COMMAND ----------

# Geographic distribution (Top 15 states)
geo_analysis = customer_demo.groupBy("state").agg(
    F.count("customer_id").alias("customer_count"),
    F.avg("annual_income").alias("avg_income"),
    F.sum(F.when(F.col("email_subscribed") == True, 1).otherwise(0)).alias("subscribed")
).orderBy(F.desc("customer_count")).limit(15)

print("\n=== TOP 15 STATES BY CUSTOMER COUNT ===")
display(geo_analysis)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 ML-Based Segmentation

# COMMAND ----------

# Use K-Means clustering for advanced segmentation
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans

# Prepare features for clustering
cluster_features = ["age", "annual_income", "recency_days"]

assembler = VectorAssembler(inputCols=cluster_features, outputCol="features")
customer_ml = assembler.transform(customer_rfm)

# Scale features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
customer_ml = scaler.fit(customer_ml).transform(customer_ml)

# Train K-Means with 5 clusters
kmeans = KMeans(featuresCol="scaled_features", k=5, seed=42)
kmeans_model = kmeans.fit(customer_ml)
customer_ml = kmeans_model.transform(customer_ml)

# Rename prediction to cluster_id
customer_ml = customer_ml.withColumnRenamed("prediction", "cluster_id")

print("\n=== ML-BASED CUSTOMER CLUSTERING ===")

# COMMAND ----------

# Analyze ML clusters
ml_cluster_analysis = customer_ml.groupBy("cluster_id").agg(
    F.count("customer_id").alias("size"),
    F.avg("age").alias("avg_age"),
    F.avg("annual_income").alias("avg_income"),
    F.avg("recency_days").alias("avg_recency"),
    F.sum(F.when(F.col("email_subscribed") == True, 1).otherwise(0)).alias("subscribed_count")
).orderBy("cluster_id")

display(ml_cluster_analysis)

# COMMAND ----------

# Visualize ML clusters
cluster_viz = customer_ml.select("age", "annual_income", "cluster_id").toPandas()

fig = px.scatter(cluster_viz, x='age', y='annual_income', color='cluster_id',
                 title='ML-Based Customer Clusters',
                 labels={'age': 'Age', 'annual_income': 'Annual Income ($)'},
                 color_continuous_scale='viridis')
fig.update_layout(height=600)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Use Case 3: Time Series Analysis
# MAGIC
# MAGIC ## Objective: Analyze temporal patterns and forecast trends
# MAGIC
# MAGIC **Focus Areas:**
# MAGIC - Web traffic trends
# MAGIC - Seasonality detection
# MAGIC - Growth rate analysis
# MAGIC - Performance forecasting

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 Web Traffic Analysis

# COMMAND ----------

# Prepare web traffic data
print("=== WEB TRAFFIC TIME SERIES ANALYSIS ===\n")

web_ts = web_traffic_df \
    .withColumn("date", F.to_date(F.col("date"))) \
    .withColumn("year", F.year(F.col("date"))) \
    .withColumn("month", F.month(F.col("date"))) \
    .withColumn("day_of_week", F.dayofweek(F.col("date"))) \
    .orderBy("date")

# Overall metrics
web_metrics = web_ts.agg(
    F.sum("page_views").alias("total_page_views"),
    F.sum("unique_visitors").alias("total_visitors"),
    F.avg("conversion_rate").alias("avg_conversion_rate"),
    F.sum("revenue").alias("total_revenue"),
    F.avg("bounce_rate").alias("avg_bounce_rate"),
    F.avg("mobile_traffic_pct").alias("avg_mobile_pct")
).collect()[0]

print(f"Total Page Views: {web_metrics['total_page_views']:,}")
print(f"Total Unique Visitors: {web_metrics['total_visitors']:,}")
print(f"Average Conversion Rate: {web_metrics['avg_conversion_rate']:.2f}%")
print(f"Total Revenue: ${web_metrics['total_revenue']:,.2f}")
print(f"Average Bounce Rate: {web_metrics['avg_bounce_rate']:.2f}%")
print(f"Average Mobile Traffic: {web_metrics['avg_mobile_pct']:.2f}%")

# COMMAND ----------

# Time series visualization
web_ts_pd = web_ts.toPandas()

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    subplot_titles=('Page Views Over Time', 'Conversion Rate', 'Revenue'),
    vertical_spacing=0.08
)

# Page views
fig.add_trace(
    go.Scatter(x=web_ts_pd['date'], y=web_ts_pd['page_views'],
               mode='lines', name='Page Views', line=dict(color='blue')),
    row=1, col=1
)

# Conversion rate
fig.add_trace(
    go.Scatter(x=web_ts_pd['date'], y=web_ts_pd['conversion_rate'],
               mode='lines', name='Conversion Rate', line=dict(color='green')),
    row=2, col=1
)

# Revenue
fig.add_trace(
    go.Scatter(x=web_ts_pd['date'], y=web_ts_pd['revenue'],
               mode='lines', name='Revenue', line=dict(color='purple')),
    row=3, col=1
)

fig.update_layout(height=800, title_text="Web Traffic Metrics Over Time", showlegend=False)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 Trend Analysis with Moving Averages

# COMMAND ----------

# Calculate moving averages
window_7 = Window.orderBy("date").rowsBetween(-6, 0)
window_30 = Window.orderBy("date").rowsBetween(-29, 0)

web_with_ma = web_ts \
    .withColumn("page_views_ma7", F.avg("page_views").over(window_7)) \
    .withColumn("page_views_ma30", F.avg("page_views").over(window_30)) \
    .withColumn("revenue_ma7", F.avg("revenue").over(window_7)) \
    .withColumn("revenue_ma30", F.avg("revenue").over(window_30)) \
    .withColumn("conversion_ma7", F.avg("conversion_rate").over(window_7))

display(web_with_ma.orderBy(F.desc("date")).limit(30))

# COMMAND ----------

# Visualize trends with moving averages
ma_viz = web_with_ma.toPandas()

fig = go.Figure()

# Actual page views
fig.add_trace(go.Scatter(x=ma_viz['date'], y=ma_viz['page_views'],
                         mode='lines', name='Daily Page Views',
                         line=dict(color='lightblue', width=1),
                         opacity=0.5))

# 7-day moving average
fig.add_trace(go.Scatter(x=ma_viz['date'], y=ma_viz['page_views_ma7'],
                         mode='lines', name='7-Day MA',
                         line=dict(color='blue', width=2)))

# 30-day moving average
fig.add_trace(go.Scatter(x=ma_viz['date'], y=ma_viz['page_views_ma30'],
                         mode='lines', name='30-Day MA',
                         line=dict(color='darkblue', width=3)))

fig.update_layout(
    title='Page Views with Moving Averages',
    xaxis_title='Date',
    yaxis_title='Page Views',
    height=500
)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3 Growth Rate Analysis

# COMMAND ----------

# Calculate period-over-period growth
window_lag = Window.orderBy("date")

web_growth = web_with_ma \
    .withColumn("prev_day_views", F.lag("page_views", 1).over(window_lag)) \
    .withColumn("prev_day_revenue", F.lag("revenue", 1).over(window_lag)) \
    .withColumn("views_growth_pct", 
                F.round((F.col("page_views") - F.col("prev_day_views")) / F.col("prev_day_views") * 100, 2)) \
    .withColumn("revenue_growth_pct",
                F.round((F.col("revenue") - F.col("prev_day_revenue")) / F.col("prev_day_revenue") * 100, 2))

# Calculate week-over-week growth
window_lag7 = Window.orderBy("date")

web_growth = web_growth \
    .withColumn("prev_week_views", F.lag("page_views", 7).over(window_lag7)) \
    .withColumn("wow_growth_pct",
                F.round((F.col("page_views") - F.col("prev_week_views")) / F.col("prev_week_views") * 100, 2))

print("\n=== GROWTH RATE ANALYSIS (Last 30 Days) ===")
display(web_growth.select("date", "page_views", "prev_day_views", "views_growth_pct", 
                          "revenue", "revenue_growth_pct").orderBy(F.desc("date")).limit(30))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.4 Seasonality Analysis

# COMMAND ----------

# Day of week patterns
dow_patterns = web_ts.groupBy("day_of_week").agg(
    F.avg("page_views").alias("avg_page_views"),
    F.avg("unique_visitors").alias("avg_visitors"),
    F.avg("conversion_rate").alias("avg_conversion"),
    F.avg("revenue").alias("avg_revenue")
).orderBy("day_of_week")

# Add day names
dow_patterns = dow_patterns.withColumn("day_name",
    F.when(F.col("day_of_week") == 1, "Sunday")
     .when(F.col("day_of_week") == 2, "Monday")
     .when(F.col("day_of_week") == 3, "Tuesday")
     .when(F.col("day_of_week") == 4, "Wednesday")
     .when(F.col("day_of_week") == 5, "Thursday")
     .when(F.col("day_of_week") == 6, "Friday")
     .otherwise("Saturday")
)

print("\n=== DAY OF WEEK PATTERNS ===")
display(dow_patterns.select("day_name", "avg_page_views", "avg_visitors", 
                            "avg_conversion", "avg_revenue"))

# COMMAND ----------

# Visualize weekly patterns
dow_pd = dow_patterns.toPandas()

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Avg Page Views', 'Avg Visitors', 'Avg Conversion Rate', 'Avg Revenue')
)

fig.add_trace(go.Bar(x=dow_pd['day_name'], y=dow_pd['avg_page_views'], 
                     marker_color='steelblue'), row=1, col=1)
fig.add_trace(go.Bar(x=dow_pd['day_name'], y=dow_pd['avg_visitors'], 
                     marker_color='lightgreen'), row=1, col=2)
fig.add_trace(go.Bar(x=dow_pd['day_name'], y=dow_pd['avg_conversion'], 
                     marker_color='orange'), row=2, col=1)
fig.add_trace(go.Bar(x=dow_pd['day_name'], y=dow_pd['avg_revenue'], 
                     marker_color='purple'), row=2, col=2)

fig.update_xaxes(tickangle=45)
fig.update_layout(height=600, title_text="Weekly Seasonality Patterns", showlegend=False)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.5 Correlation Analysis

# COMMAND ----------

# Analyze correlations between metrics
correlation_data = web_ts.select(
    "page_views", "unique_visitors", "sessions", 
    "bounce_rate", "conversion_rate", "revenue", "mobile_traffic_pct"
).toPandas()

# Calculate correlation matrix
correlation_matrix = correlation_data.corr()

print("\n=== CORRELATION MATRIX ===")
print(correlation_matrix)

# Visualize correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, fmt='.2f')
plt.title('Web Traffic Metrics Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.6 Conversion Funnel Analysis

# COMMAND ----------

# Calculate funnel metrics
funnel_metrics = web_ts.agg(
    F.sum("page_views").alias("total_page_views"),
    F.sum("unique_visitors").alias("total_visitors"),
    F.sum("sessions").alias("total_sessions"),
    (F.sum("sessions") * F.avg("conversion_rate") / 100).alias("total_conversions")
).collect()[0]

funnel_data = {
    'Stage': ['Page Views', 'Unique Visitors', 'Sessions', 'Conversions'],
    'Count': [
        funnel_metrics['total_page_views'],
        funnel_metrics['total_visitors'],
        funnel_metrics['total_sessions'],
        funnel_metrics['total_conversions']
    ]
}

print("\n=== CONVERSION FUNNEL ===")
for stage, count in zip(funnel_data['Stage'], funnel_data['Count']):
    print(f"{stage}: {count:,.0f}")

# Calculate conversion rates between stages
print("\n=== CONVERSION RATES ===")
print(f"Visitor to Session Rate: {funnel_metrics['total_sessions']/funnel_metrics['total_visitors']*100:.2f}%")
print(f"Session to Conversion Rate: {funnel_metrics['total_conversions']/funnel_metrics['total_sessions']*100:.2f}%")
print(f"Overall Conversion Rate: {funnel_metrics['total_conversions']/funnel_metrics['total_visitors']*100:.2f}%")

# COMMAND ----------

# Visualize funnel
fig = go.Figure(go.Funnel(
    y=funnel_data['Stage'],
    x=funnel_data['Count'],
    textinfo="value+percent previous",
    marker=dict(color=["lightblue", "lightgreen", "lightyellow", "lightcoral"])
))

fig.update_layout(
    title='Web Traffic Conversion Funnel',
    height=500
)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Summary: Actionable Insights
# MAGIC
# MAGIC ## Key Findings and Recommendations

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sales Analytics Insights
# MAGIC
# MAGIC **Top Performers:**
# MAGIC - Identify your best products, regions, and sales representatives
# MAGIC - Replicate successful strategies across underperforming areas
# MAGIC - Focus marketing resources on high-performing products
# MAGIC
# MAGIC **Temporal Patterns:**
# MAGIC - Understand daily, weekly, and seasonal trends
# MAGIC - Optimize inventory and staffing based on demand patterns
# MAGIC - Plan promotions during high-traffic periods
# MAGIC
# MAGIC **Actions:**
# MAGIC 1. Allocate more inventory to top-performing products
# MAGIC 2. Train underperforming sales reps using best practices from top performers
# MAGIC 3. Launch targeted campaigns in underperforming regions
# MAGIC 4. Adjust pricing strategies based on demand patterns

# COMMAND ----------

# MAGIC %md
# MAGIC ## Customer Segmentation Insights
# MAGIC
# MAGIC **RFM Segments:**
# MAGIC - **Champions**: Your best customers - focus on retention and referrals
# MAGIC - **At Risk**: Customers who need re-engagement campaigns
# MAGIC - **Lost**: Target with win-back campaigns
# MAGIC
# MAGIC **Demographic Insights:**
# MAGIC - Tailor marketing messages to different age groups and income brackets
# MAGIC - Customize product offerings for each segment
# MAGIC - Personalize communication based on preferences
# MAGIC
# MAGIC **Actions:**
# MAGIC 1. Create VIP program for Champions
# MAGIC 2. Send personalized re-engagement emails to At Risk customers
# MAGIC 3. Design win-back campaigns for Lost customers
# MAGIC 4. Develop age-specific product bundles
# MAGIC 5. Implement income-based pricing tiers

# COMMAND ----------

# MAGIC %md
# MAGIC ## Time Series Analysis Insights
# MAGIC
# MAGIC **Web Traffic Trends:**
# MAGIC - Monitor moving averages to identify long-term trends
# MAGIC - Track growth rates to measure success of initiatives
# MAGIC - Understand weekly patterns to optimize content publishing
# MAGIC
# MAGIC **Conversion Optimization:**
# MAGIC - Identify bottlenecks in the conversion funnel
# MAGIC - Focus on improving bounce rate on high-traffic days
# MAGIC - Optimize mobile experience (significant traffic percentage)
# MAGIC
# MAGIC **Actions:**
# MAGIC 1. Publish important content on high-traffic days
# MAGIC 2. A/B test landing pages to improve conversion rates
# MAGIC 3. Enhance mobile user experience
# MAGIC 4. Implement retargeting campaigns for bounced visitors
# MAGIC 5. Set up alerts for significant traffic drops

# COMMAND ----------

# MAGIC %md
# MAGIC # Next Steps & Best Practices
# MAGIC
# MAGIC ## 1. Automate Analysis
# MAGIC - Schedule notebooks to run daily/weekly
# MAGIC - Set up email alerts for key metrics
# MAGIC - Create automated reports for stakeholders
# MAGIC
# MAGIC ## 2. Build Dashboards
# MAGIC - Use Databricks SQL for interactive dashboards
# MAGIC - Create executive summaries with key KPIs
# MAGIC - Enable self-service analytics for business users
# MAGIC
# MAGIC ## 3. Implement ML Models
# MAGIC - Deploy predictive models to production
# MAGIC - Monitor model performance over time
# MAGIC - Retrain models with new data regularly
# MAGIC
# MAGIC ## 4. Data Quality
# MAGIC - Establish data quality checks
# MAGIC - Monitor data pipelines for failures
# MAGIC - Document data definitions and lineage
# MAGIC
# MAGIC ## 5. Collaboration
# MAGIC - Share notebooks with team members
# MAGIC - Use version control for code
# MAGIC - Document insights and decisions

# COMMAND ----------

# MAGIC %md
# MAGIC # Congratulations!
# MAGIC
# MAGIC You've completed all 6 Databricks tutorial notebooks covering:
# MAGIC
# MAGIC 1. ✅ Data Exploration & Analysis
# MAGIC 2. ✅ Data Visualization
# MAGIC 3. ✅ SQL Analytics
# MAGIC 4. ✅ ETL & Data Processing
# MAGIC 5. ✅ Machine Learning
# MAGIC 6. ✅ Real-World Use Cases
# MAGIC
# MAGIC ## You've Learned:
# MAGIC - Loading and exploring data in Databricks
# MAGIC - Data cleaning and transformation techniques
# MAGIC - Creating visualizations with multiple libraries
# MAGIC - Writing SQL queries and building data warehouses
# MAGIC - Building ETL pipelines
# MAGIC - Training and evaluating ML models
# MAGIC - Applying analytics to real business problems
# MAGIC
# MAGIC ## Continue Your Journey:
# MAGIC - Explore Databricks documentation at docs.databricks.com
# MAGIC - Join the Databricks community
# MAGIC - Practice with your own datasets
# MAGIC - Build end-to-end data projects
# MAGIC
# MAGIC **Happy Data Engineering and Analysis!** 🚀
