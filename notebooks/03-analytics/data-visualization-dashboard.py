# Databricks notebook source
# MAGIC %md
# MAGIC # Interactive Data Visualization and Dashboards
# MAGIC 
# MAGIC This notebook demonstrates advanced data visualization techniques in Databricks:
# MAGIC - Built-in visualization capabilities
# MAGIC - Interactive charts and dashboards
# MAGIC - Advanced SQL analytics with visualizations
# MAGIC - Performance monitoring dashboards
# MAGIC 
# MAGIC **Business Value**: Create compelling data stories and interactive dashboards
# MAGIC **Skills**: Data visualization, dashboard design, advanced SQL analytics

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Data Preparation

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import random
from datetime import datetime, timedelta

# Create comprehensive sample data for visualization
def create_sales_dashboard_data():
    """Create sample sales data for dashboard demonstration"""
    
    # Product categories and regions
    categories = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books", "Beauty"]
    regions = ["North", "South", "East", "West", "Central"]
    sales_reps = ["Alice Johnson", "Bob Smith", "Carol Wilson", "David Brown", "Eva Davis"]
    
    # Generate sales data for the last 2 years
    sales_data = []
    start_date = datetime.now() - timedelta(days=730)
    
    for i in range(10000):  # 10K sales records
        sale_date = start_date + timedelta(days=random.randint(0, 730))
        
        category = random.choice(categories)
        region = random.choice(regions)
        sales_rep = random.choice(sales_reps)
        
        # Seasonal adjustments
        month = sale_date.month
        seasonal_multiplier = 1.0
        if 11 <= month <= 12:  # Holiday season
            seasonal_multiplier = 1.5
        elif 6 <= month <= 8:   # Summer
            seasonal_multiplier = 1.2
        
        # Regional adjustments
        regional_multiplier = {"North": 1.2, "South": 0.9, "East": 1.1, "West": 1.3, "Central": 1.0}[region]
        
        base_amount = random.uniform(50, 2000)
        amount = base_amount * seasonal_multiplier * regional_multiplier
        
        quantity = random.randint(1, 10)
        discount = random.choice([0, 0.05, 0.10, 0.15, 0.20])
        final_amount = amount * (1 - discount)
        
        sales_data.append((
            f"SALE_{i+1:06d}",
            sale_date,
            category,
            region,
            sales_rep,
            round(amount, 2),
            quantity,
            discount,
            round(final_amount, 2)
        ))
    
    return sales_data

# Generate the data
sales_records = create_sales_dashboard_data()

# Create DataFrame
sales_schema = StructType([
    StructField("sale_id", StringType(), True),
    StructField("sale_date", TimestampType(), True),
    StructField("category", StringType(), True),
    StructField("region", StringType(), True),
    StructField("sales_rep", StringType(), True),
    StructField("gross_amount", DoubleType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("discount", DoubleType(), True),
    StructField("net_amount", DoubleType(), True)
])

sales_df = spark.createDataFrame(sales_records, sales_schema)

# Add time-based features for better analytics
sales_enriched = sales_df.withColumn("year", year(col("sale_date"))) \
    .withColumn("month", month(col("sale_date"))) \
    .withColumn("quarter", quarter(col("sale_date"))) \
    .withColumn("day_of_week", dayofweek(col("sale_date"))) \
    .withColumn("week_of_year", weekofyear(col("sale_date"))) \
    .withColumn("is_weekend", when(col("day_of_week").isin([1, 7]), 1).otherwise(0))

# Cache for better performance
sales_enriched.cache()
print(f"Created sales dataset with {sales_enriched.count():,} records")

# Create a view for SQL access
sales_enriched.createOrReplaceTempView("sales_data")

display(sales_enriched.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Executive Summary Dashboard
# MAGIC 
# MAGIC Key metrics and KPIs for executive overview

# COMMAND ----------

# Calculate key metrics
summary_metrics = sales_enriched.agg(
    sum("net_amount").alias("total_revenue"),
    count("sale_id").alias("total_transactions"),
    countDistinct("sales_rep").alias("active_sales_reps"),
    avg("net_amount").alias("avg_transaction_value"),
    sum("quantity").alias("total_units_sold")
).collect()[0]

print("=== EXECUTIVE DASHBOARD ===")
print(f"📊 Total Revenue: ${summary_metrics['total_revenue']:,.2f}")
print(f"🛒 Total Transactions: {summary_metrics['total_transactions']:,}")
print(f"👥 Active Sales Reps: {summary_metrics['active_sales_reps']}")
print(f"💰 Average Transaction: ${summary_metrics['avg_transaction_value']:.2f}")
print(f"📦 Total Units Sold: {summary_metrics['total_units_sold']:,}")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Monthly Revenue Trend
# MAGIC SELECT 
# MAGIC   CONCAT(year, '-', LPAD(month, 2, '0')) as month_year,
# MAGIC   SUM(net_amount) as monthly_revenue,
# MAGIC   COUNT(*) as transactions,
# MAGIC   AVG(net_amount) as avg_transaction_value
# MAGIC FROM sales_data
# MAGIC GROUP BY year, month
# MAGIC ORDER BY year, month

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📈 Click the chart icon above to create visualizations!
# MAGIC 
# MAGIC **Try these visualizations:**
# MAGIC 1. **Line Chart**: monthly_revenue over month_year (shows trend)
# MAGIC 2. **Bar Chart**: Compare monthly_revenue across different months
# MAGIC 3. **Combo Chart**: Revenue (bar) + Transaction count (line)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Regional Performance Analysis

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Regional Performance Comparison
# MAGIC SELECT 
# MAGIC   region,
# MAGIC   SUM(net_amount) as total_revenue,
# MAGIC   COUNT(*) as transactions,
# MAGIC   AVG(net_amount) as avg_transaction,
# MAGIC   SUM(quantity) as units_sold,
# MAGIC   COUNT(DISTINCT sales_rep) as sales_reps
# MAGIC FROM sales_data
# MAGIC GROUP BY region
# MAGIC ORDER BY total_revenue DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Regional Performance by Quarter
# MAGIC SELECT 
# MAGIC   region,
# MAGIC   CONCAT('Q', quarter, ' ', year) as quarter_year,
# MAGIC   SUM(net_amount) as quarterly_revenue,
# MAGIC   COUNT(*) as transactions
# MAGIC FROM sales_data
# MAGIC GROUP BY region, year, quarter
# MAGIC ORDER BY year, quarter, region

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Product Category Analysis

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Category Performance with Growth Analysis
# MAGIC WITH category_monthly AS (
# MAGIC   SELECT 
# MAGIC     category,
# MAGIC     year,
# MAGIC     month,
# MAGIC     SUM(net_amount) as monthly_revenue
# MAGIC   FROM sales_data
# MAGIC   GROUP BY category, year, month
# MAGIC ),
# MAGIC category_growth AS (
# MAGIC   SELECT 
# MAGIC     *,
# MAGIC     LAG(monthly_revenue) OVER (
# MAGIC       PARTITION BY category 
# MAGIC       ORDER BY year, month
# MAGIC     ) as prev_month_revenue,
# MAGIC     monthly_revenue - LAG(monthly_revenue) OVER (
# MAGIC       PARTITION BY category 
# MAGIC       ORDER BY year, month
# MAGIC     ) as revenue_change
# MAGIC   FROM category_monthly
# MAGIC )
# MAGIC 
# MAGIC SELECT 
# MAGIC   category,
# MAGIC   CONCAT(year, '-', LPAD(month, 2, '0')) as month_year,
# MAGIC   monthly_revenue,
# MAGIC   revenue_change,
# MAGIC   CASE 
# MAGIC     WHEN prev_month_revenue > 0 
# MAGIC     THEN ROUND((revenue_change / prev_month_revenue) * 100, 2)
# MAGIC     ELSE 0 
# MAGIC   END as growth_rate_percent
# MAGIC FROM category_growth
# MAGIC WHERE prev_month_revenue IS NOT NULL
# MAGIC ORDER BY year, month, category

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Category Market Share Analysis
# MAGIC SELECT 
# MAGIC   category,
# MAGIC   SUM(net_amount) as category_revenue,
# MAGIC   ROUND(
# MAGIC     SUM(net_amount) * 100.0 / SUM(SUM(net_amount)) OVER (), 2
# MAGIC   ) as market_share_percent,
# MAGIC   COUNT(*) as transactions,
# MAGIC   AVG(net_amount) as avg_transaction_value
# MAGIC FROM sales_data
# MAGIC GROUP BY category
# MAGIC ORDER BY category_revenue DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Sales Representative Performance

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Sales Rep Leaderboard
# MAGIC SELECT 
# MAGIC   sales_rep,
# MAGIC   SUM(net_amount) as total_sales,
# MAGIC   COUNT(*) as transactions,
# MAGIC   AVG(net_amount) as avg_deal_size,
# MAGIC   SUM(quantity) as units_sold,
# MAGIC   ROUND(AVG(discount) * 100, 1) as avg_discount_percent,
# MAGIC   COUNT(DISTINCT category) as categories_sold
# MAGIC FROM sales_data
# MAGIC GROUP BY sales_rep
# MAGIC ORDER BY total_sales DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Sales Rep Performance Trends
# MAGIC SELECT 
# MAGIC   sales_rep,
# MAGIC   CONCAT(year, '-', LPAD(month, 2, '0')) as month_year,
# MAGIC   SUM(net_amount) as monthly_sales,
# MAGIC   COUNT(*) as monthly_transactions,
# MAGIC   RANK() OVER (
# MAGIC     PARTITION BY year, month 
# MAGIC     ORDER BY SUM(net_amount) DESC
# MAGIC   ) as monthly_rank
# MAGIC FROM sales_data
# MAGIC GROUP BY sales_rep, year, month
# MAGIC ORDER BY year, month, monthly_sales DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Advanced Analytics Dashboard

# COMMAND ----------

# Customer segmentation based on purchase behavior
customer_segments = sales_enriched.groupBy("sales_rep").agg(
    sum("net_amount").alias("total_sales"),
    count("sale_id").alias("transaction_count"),
    avg("net_amount").alias("avg_transaction"),
    stddev("net_amount").alias("sales_volatility")
).withColumn(
    "performance_tier",
    when(col("total_sales") >= 100000, "Top Performer")
    .when(col("total_sales") >= 50000, "High Performer")
    .when(col("total_sales") >= 25000, "Average Performer")
    .otherwise("Developing")
)

display(customer_segments.orderBy(col("total_sales").desc()))

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Seasonal Analysis
# MAGIC SELECT 
# MAGIC   CASE 
# MAGIC     WHEN month IN (12, 1, 2) THEN 'Winter'
# MAGIC     WHEN month IN (3, 4, 5) THEN 'Spring'
# MAGIC     WHEN month IN (6, 7, 8) THEN 'Summer'
# MAGIC     ELSE 'Fall'
# MAGIC   END as season,
# MAGIC   category,
# MAGIC   SUM(net_amount) as seasonal_revenue,
# MAGIC   COUNT(*) as transactions,
# MAGIC   AVG(net_amount) as avg_transaction
# MAGIC FROM sales_data
# MAGIC GROUP BY 
# MAGIC   CASE 
# MAGIC     WHEN month IN (12, 1, 2) THEN 'Winter'
# MAGIC     WHEN month IN (3, 4, 5) THEN 'Spring'
# MAGIC     WHEN month IN (6, 7, 8) THEN 'Summer'
# MAGIC     ELSE 'Fall'
# MAGIC   END,
# MAGIC   category
# MAGIC ORDER BY season, seasonal_revenue DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Real-time Performance Monitoring

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Daily Performance Metrics (Last 30 Days Simulation)
# MAGIC WITH daily_metrics AS (
# MAGIC   SELECT 
# MAGIC     DATE(sale_date) as sale_date,
# MAGIC     SUM(net_amount) as daily_revenue,
# MAGIC     COUNT(*) as daily_transactions,
# MAGIC     COUNT(DISTINCT sales_rep) as active_reps,
# MAGIC     AVG(net_amount) as avg_transaction_value
# MAGIC   FROM sales_data
# MAGIC   WHERE sale_date >= DATE_SUB(CURRENT_DATE(), 90)  -- Last 90 days
# MAGIC   GROUP BY DATE(sale_date)
# MAGIC ),
# MAGIC metrics_with_targets AS (
# MAGIC   SELECT 
# MAGIC     *,
# MAGIC     10000.0 as daily_revenue_target,  -- $10K daily target
# MAGIC     50 as daily_transaction_target,   -- 50 transactions target
# MAGIC     CASE 
# MAGIC       WHEN daily_revenue >= 10000 THEN 'Above Target'
# MAGIC       WHEN daily_revenue >= 8000 THEN 'Near Target'
# MAGIC       ELSE 'Below Target'
# MAGIC     END as performance_status
# MAGIC   FROM daily_metrics
# MAGIC )
# MAGIC 
# MAGIC SELECT 
# MAGIC   sale_date,
# MAGIC   daily_revenue,
# MAGIC   daily_revenue_target,
# MAGIC   daily_transactions,
# MAGIC   performance_status,
# MAGIC   ROUND((daily_revenue / daily_revenue_target) * 100, 1) as target_achievement_percent
# MAGIC FROM metrics_with_targets
# MAGIC ORDER BY sale_date DESC
# MAGIC LIMIT 30

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Interactive Drill-Down Analysis

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Multi-dimensional Analysis: Region, Category, Time
# MAGIC SELECT 
# MAGIC   region,
# MAGIC   category,
# MAGIC   CONCAT('Q', quarter, ' ', year) as quarter_year,
# MAGIC   SUM(net_amount) as revenue,
# MAGIC   COUNT(*) as transactions,
# MAGIC   AVG(net_amount) as avg_transaction,
# MAGIC   SUM(quantity) as units_sold,
# MAGIC   
# MAGIC   -- Performance vs. overall average
# MAGIC   ROUND(
# MAGIC     AVG(net_amount) / AVG(AVG(net_amount)) OVER () * 100, 1
# MAGIC   ) as performance_index,
# MAGIC   
# MAGIC   -- Ranking within region
# MAGIC   RANK() OVER (
# MAGIC     PARTITION BY region, year, quarter 
# MAGIC     ORDER BY SUM(net_amount) DESC
# MAGIC   ) as category_rank_in_region
# MAGIC   
# MAGIC FROM sales_data
# MAGIC GROUP BY region, category, year, quarter
# MAGIC ORDER BY year, quarter, region, revenue DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Custom Visualization Examples

# COMMAND ----------

# Create data for custom visualizations
# Cohort analysis for visualization
cohort_data = spark.sql("""
WITH first_purchase AS (
  SELECT 
    sales_rep,
    MIN(DATE_TRUNC('month', sale_date)) as cohort_month
  FROM sales_data
  GROUP BY sales_rep
),
purchase_activity AS (
  SELECT 
    fp.sales_rep,
    fp.cohort_month,
    DATE_TRUNC('month', sd.sale_date) as purchase_month,
    MONTHS_BETWEEN(DATE_TRUNC('month', sd.sale_date), fp.cohort_month) as period_number
  FROM first_purchase fp
  JOIN sales_data sd ON fp.sales_rep = sd.sales_rep
)

SELECT 
  cohort_month,
  period_number,
  COUNT(DISTINCT sales_rep) as active_reps
FROM purchase_activity
GROUP BY cohort_month, period_number
ORDER BY cohort_month, period_number
""")

print("Cohort Analysis Data:")
display(cohort_data)

# COMMAND ----------

# Geographic performance simulation
geographic_performance = spark.sql("""
SELECT 
  region,
  category,
  SUM(net_amount) as revenue,
  COUNT(*) as transactions,
  
  -- Coordinates for map visualization (simulated)
  CASE region
    WHEN 'North' THEN 45.0
    WHEN 'South' THEN 30.0  
    WHEN 'East' THEN 40.0
    WHEN 'West' THEN 38.0
    ELSE 39.0
  END as latitude,
  
  CASE region
    WHEN 'North' THEN -95.0
    WHEN 'South' THEN -85.0
    WHEN 'East' THEN -75.0  
    WHEN 'West' THEN -120.0
    ELSE -95.0
  END as longitude
  
FROM sales_data
GROUP BY region, category
ORDER BY region, revenue DESC
""")

print("Geographic Performance Data:")
display(geographic_performance)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Dashboard Summary and Insights

# COMMAND ----------

# Generate final insights summary
insights_summary = spark.sql("""
WITH monthly_trends AS (
  SELECT 
    year,
    month,
    SUM(net_amount) as monthly_revenue,
    LAG(SUM(net_amount)) OVER (ORDER BY year, month) as prev_month_revenue
  FROM sales_data
  GROUP BY year, month
),
performance_insights AS (
  SELECT 
    COUNT(*) as total_months,
    SUM(CASE WHEN monthly_revenue > prev_month_revenue THEN 1 ELSE 0 END) as growth_months,
    AVG(monthly_revenue) as avg_monthly_revenue,
    MAX(monthly_revenue) as peak_monthly_revenue,
    MIN(monthly_revenue) as lowest_monthly_revenue
  FROM monthly_trends
  WHERE prev_month_revenue IS NOT NULL
)

SELECT 
  total_months,
  growth_months,
  ROUND(growth_months * 100.0 / total_months, 1) as growth_month_percentage,
  ROUND(avg_monthly_revenue, 2) as avg_monthly_revenue,
  ROUND(peak_monthly_revenue, 2) as peak_monthly_revenue,
  ROUND(lowest_monthly_revenue, 2) as lowest_monthly_revenue,
  ROUND((peak_monthly_revenue - lowest_monthly_revenue) / avg_monthly_revenue * 100, 1) as revenue_volatility_percent
FROM performance_insights
""")

print("📊 BUSINESS INSIGHTS SUMMARY:")
display(insights_summary)

# COMMAND ----------

# Top performing combinations
top_combinations = spark.sql("""
SELECT 
  region,
  category,
  sales_rep,
  SUM(net_amount) as combination_revenue,
  COUNT(*) as transactions,
  AVG(net_amount) as avg_deal_size,
  RANK() OVER (ORDER BY SUM(net_amount) DESC) as overall_rank
FROM sales_data
GROUP BY region, category, sales_rep
HAVING COUNT(*) >= 10  -- Minimum 10 transactions for significance
ORDER BY combination_revenue DESC
LIMIT 15
""")

print("🏆 TOP PERFORMING COMBINATIONS:")
display(top_combinations)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Insights and Recommendations
# MAGIC 
# MAGIC ### 📊 Visualization Best Practices Demonstrated:
# MAGIC 
# MAGIC 1. **Executive Dashboard**: High-level KPIs and metrics
# MAGIC 2. **Trend Analysis**: Time-series charts for pattern identification
# MAGIC 3. **Comparative Analysis**: Regional and category performance
# MAGIC 4. **Drill-down Capability**: Multi-dimensional analysis
# MAGIC 5. **Performance Monitoring**: Target vs. actual comparisons
# MAGIC 
# MAGIC ### 🎨 Databricks Visualization Features:
# MAGIC 
# MAGIC - **Built-in Charts**: Line, bar, pie, scatter, and combination charts
# MAGIC - **Interactive Elements**: Hover details, zoom, and filtering
# MAGIC - **Dashboard Creation**: Combine multiple visualizations
# MAGIC - **Export Options**: PNG, PDF, and data export capabilities
# MAGIC - **Real-time Updates**: Refresh data automatically
# MAGIC 
# MAGIC ### 📈 Business Intelligence Capabilities:
# MAGIC 
# MAGIC - **KPI Monitoring**: Track key business metrics
# MAGIC - **Trend Analysis**: Identify patterns and seasonality
# MAGIC - **Performance Comparison**: Benchmark across dimensions
# MAGIC - **Anomaly Detection**: Spot unusual patterns
# MAGIC - **Predictive Insights**: Forecast future performance
# MAGIC 
# MAGIC ### 🚀 Advanced Features to Explore:
# MAGIC 
# MAGIC 1. **Real-time Dashboards**: Connect to streaming data
# MAGIC 2. **Parameterized Reports**: Dynamic filtering and selection
# MAGIC 3. **Automated Alerting**: Notifications for threshold breaches
# MAGIC 4. **Mobile Optimization**: Responsive dashboard design
# MAGIC 5. **Integration**: Embed dashboards in external applications
# MAGIC 
# MAGIC ### 💡 Next Steps:
# MAGIC 
# MAGIC - **Create Custom Dashboards**: Combine charts into comprehensive views
# MAGIC - **Schedule Reports**: Automate report generation and distribution
# MAGIC - **Add Interactivity**: Implement filters and drill-down capabilities
# MAGIC - **Performance Optimization**: Optimize queries for faster rendering
# MAGIC - **User Access Control**: Set up permissions and sharing

# COMMAND ----------

# Cleanup
spark.catalog.clearCache()
print("✅ Dashboard demo completed successfully!")
print("💡 Remember to create visualizations by clicking the chart icon after running SQL queries!")

# COMMAND ----------