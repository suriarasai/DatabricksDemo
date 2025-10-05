# Databricks notebook source
# MAGIC %md
# MAGIC # Tutorial 3: SQL Analytics in Databricks
# MAGIC
# MAGIC ## Overview
# MAGIC This notebook covers SQL operations in Databricks:
# MAGIC - SQL Fundamentals with Spark SQL
# MAGIC - Advanced SQL techniques (Window functions, CTEs)
# MAGIC - Data warehousing concepts (Tables, Views, Optimization)
# MAGIC
# MAGIC **Key Concepts:**
# MAGIC - Spark SQL: Query DataFrames using SQL syntax
# MAGIC - Temp Views: Create temporary SQL-queryable tables
# MAGIC - Persistent Tables: Store data in Delta Lake format
# MAGIC - Query Optimization: Improve query performance

# COMMAND ----------

# Import libraries
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Load datasets
customer_df = spark.read.csv("/Volumes/workspace/sample/datasets/customer_data.csv", header=True, inferSchema=True)
products_df = spark.read.csv("/Volumes/workspace/sample/datasets/products.csv", header=True, inferSchema=True)
sales_df = spark.read.csv("/Volumes/workspace/sample/datasets/sales_data.csv", header=True, inferSchema=True)
web_traffic_df = spark.read.csv("/Volumes/workspace/sample/datasets/web_traffic.csv", header=True, inferSchema=True)

print("Datasets loaded successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. SQL Fundamentals
# MAGIC
# MAGIC ### Creating Temporary Views
# MAGIC
# MAGIC **Concept:** Temporary views allow you to query DataFrames using SQL
# MAGIC
# MAGIC **Syntax:**
# MAGIC - `.createOrReplaceTempView("view_name")`: Session-scoped view
# MAGIC - `.createOrReplaceGlobalTempView("view_name")`: Global view (access with `global_temp.view_name`)

# COMMAND ----------

# Create temporary views for SQL queries
customer_df.createOrReplaceTempView("customers")
products_df.createOrReplaceTempView("products")
sales_df.createOrReplaceTempView("sales")
web_traffic_df.createOrReplaceTempView("web_traffic")

print("Temporary views created: customers, products, sales, web_traffic")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Basic SELECT Queries
# MAGIC
# MAGIC **SQL Syntax:**
# MAGIC ```sql
# MAGIC SELECT column1, column2
# MAGIC FROM table_name
# MAGIC WHERE condition
# MAGIC ORDER BY column
# MAGIC LIMIT n
# MAGIC ```

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Simple SELECT: View first 10 customers
# MAGIC SELECT customer_id, first_name, last_name, email, city, state
# MAGIC FROM customers
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %sql
# MAGIC -- SELECT with WHERE clause: Filter customers by state
# MAGIC SELECT customer_id, first_name, last_name, city, annual_income
# MAGIC FROM customers
# MAGIC WHERE state = 'California'
# MAGIC ORDER BY annual_income DESC
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %sql
# MAGIC -- SELECT with multiple conditions
# MAGIC SELECT customer_id, first_name, last_name, age, annual_income, segment
# MAGIC FROM customers
# MAGIC WHERE age > 30 
# MAGIC   AND annual_income > 50000
# MAGIC   AND email_subscribed = true
# MAGIC ORDER BY annual_income DESC
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %md
# MAGIC ### Aggregate Functions
# MAGIC
# MAGIC **Common Aggregations:**
# MAGIC - `COUNT()`: Count rows
# MAGIC - `SUM()`: Sum values
# MAGIC - `AVG()`: Average values
# MAGIC - `MIN()` / `MAX()`: Minimum / Maximum values
# MAGIC - `STDDEV()`: Standard deviation

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Aggregate functions: Customer statistics
# MAGIC SELECT 
# MAGIC   COUNT(*) as total_customers,
# MAGIC   AVG(age) as avg_age,
# MAGIC   AVG(annual_income) as avg_income,
# MAGIC   MIN(annual_income) as min_income,
# MAGIC   MAX(annual_income) as max_income,
# MAGIC   STDDEV(annual_income) as stddev_income
# MAGIC FROM customers

# COMMAND ----------

# MAGIC %md
# MAGIC ### GROUP BY
# MAGIC
# MAGIC **Purpose:** Aggregate data by categories
# MAGIC
# MAGIC **Syntax:**
# MAGIC ```sql
# MAGIC SELECT column, AGG_FUNCTION(column)
# MAGIC FROM table
# MAGIC GROUP BY column
# MAGIC HAVING condition
# MAGIC ```

# COMMAND ----------

# MAGIC %sql
# MAGIC -- GROUP BY: Customer distribution by segment
# MAGIC SELECT 
# MAGIC   segment,
# MAGIC   COUNT(*) as customer_count,
# MAGIC   AVG(age) as avg_age,
# MAGIC   AVG(annual_income) as avg_income,
# MAGIC   SUM(CASE WHEN email_subscribed THEN 1 ELSE 0 END) as subscribed_count
# MAGIC FROM customers
# MAGIC GROUP BY segment
# MAGIC ORDER BY customer_count DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- GROUP BY with HAVING: States with high-income customers
# MAGIC SELECT 
# MAGIC   state,
# MAGIC   COUNT(*) as customer_count,
# MAGIC   AVG(annual_income) as avg_income
# MAGIC FROM customers
# MAGIC GROUP BY state
# MAGIC HAVING AVG(annual_income) > 60000
# MAGIC ORDER BY avg_income DESC
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ### JOINs
# MAGIC
# MAGIC **Types of Joins:**
# MAGIC - `INNER JOIN`: Returns matching rows from both tables
# MAGIC - `LEFT JOIN`: All rows from left table + matching from right
# MAGIC - `RIGHT JOIN`: All rows from right table + matching from left
# MAGIC - `FULL OUTER JOIN`: All rows from both tables

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Note: Since our sales table has product names (not IDs), 
# MAGIC -- we'll create a simulated join scenario
# MAGIC
# MAGIC -- Sales by region with aggregated metrics
# MAGIC SELECT 
# MAGIC   s.region,
# MAGIC   COUNT(DISTINCT s.transaction_id) as total_transactions,
# MAGIC   COUNT(DISTINCT s.product) as unique_products,
# MAGIC   SUM(s.total_sales) as total_revenue,
# MAGIC   AVG(s.customer_satisfaction) as avg_satisfaction,
# MAGIC   SUM(s.quantity) as total_units_sold
# MAGIC FROM sales s
# MAGIC GROUP BY s.region
# MAGIC ORDER BY total_revenue DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Product analysis with category grouping
# MAGIC SELECT 
# MAGIC   category,
# MAGIC   COUNT(*) as product_count,
# MAGIC   AVG(price) as avg_price,
# MAGIC   AVG(rating) as avg_rating,
# MAGIC   SUM(num_reviews) as total_reviews,
# MAGIC   MIN(price) as min_price,
# MAGIC   MAX(price) as max_price
# MAGIC FROM products
# MAGIC GROUP BY category
# MAGIC ORDER BY avg_rating DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Advanced SQL Techniques
# MAGIC
# MAGIC ### Window Functions
# MAGIC
# MAGIC **Purpose:** Perform calculations across rows related to current row
# MAGIC
# MAGIC **Common Window Functions:**
# MAGIC - `ROW_NUMBER()`: Assigns unique row number
# MAGIC - `RANK()` / `DENSE_RANK()`: Ranking with or without gaps
# MAGIC - `LAG()` / `LEAD()`: Access previous/next row values
# MAGIC - `SUM()`, `AVG()` with OVER: Running totals/averages

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ROW_NUMBER: Rank customers by income within each segment
# MAGIC SELECT 
# MAGIC   customer_id,
# MAGIC   first_name,
# MAGIC   last_name,
# MAGIC   segment,
# MAGIC   annual_income,
# MAGIC   ROW_NUMBER() OVER (PARTITION BY segment ORDER BY annual_income DESC) as income_rank
# MAGIC FROM customers
# MAGIC QUALIFY income_rank <= 5
# MAGIC ORDER BY segment, income_rank

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Running total: Cumulative sales over time
# MAGIC WITH daily_sales AS (
# MAGIC   SELECT 
# MAGIC     date,
# MAGIC     SUM(total_sales) as daily_revenue
# MAGIC   FROM sales
# MAGIC   GROUP BY date
# MAGIC   ORDER BY date
# MAGIC )
# MAGIC SELECT 
# MAGIC   date,
# MAGIC   daily_revenue,
# MAGIC   SUM(daily_revenue) OVER (ORDER BY date) as cumulative_revenue,
# MAGIC   AVG(daily_revenue) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as moving_avg_7days
# MAGIC FROM daily_sales
# MAGIC ORDER BY date

# COMMAND ----------

# MAGIC %sql
# MAGIC -- LAG function: Compare with previous period
# MAGIC WITH daily_metrics AS (
# MAGIC   SELECT 
# MAGIC     date,
# MAGIC     SUM(total_sales) as daily_revenue,
# MAGIC     COUNT(transaction_id) as transaction_count
# MAGIC   FROM sales
# MAGIC   GROUP BY date
# MAGIC )
# MAGIC SELECT 
# MAGIC   date,
# MAGIC   daily_revenue,
# MAGIC   LAG(daily_revenue, 1) OVER (ORDER BY date) as prev_day_revenue,
# MAGIC   daily_revenue - LAG(daily_revenue, 1) OVER (ORDER BY date) as revenue_change,
# MAGIC   ROUND(((daily_revenue - LAG(daily_revenue, 1) OVER (ORDER BY date)) / 
# MAGIC          LAG(daily_revenue, 1) OVER (ORDER BY date) * 100), 2) as pct_change
# MAGIC FROM daily_metrics
# MAGIC ORDER BY date DESC
# MAGIC LIMIT 30

# COMMAND ----------

# MAGIC %md
# MAGIC ### Common Table Expressions (CTEs)
# MAGIC
# MAGIC **Purpose:** Create temporary named result sets for complex queries
# MAGIC
# MAGIC **Syntax:**
# MAGIC ```sql
# MAGIC WITH cte_name AS (
# MAGIC   SELECT ...
# MAGIC )
# MAGIC SELECT * FROM cte_name
# MAGIC ```
# MAGIC
# MAGIC **Benefits:**
# MAGIC - Improves query readability
# MAGIC - Can reference multiple times
# MAGIC - Useful for breaking down complex logic

# COMMAND ----------

# MAGIC %sql
# MAGIC -- CTE Example: Multi-step analysis
# MAGIC WITH customer_stats AS (
# MAGIC   SELECT 
# MAGIC     segment,
# MAGIC     COUNT(*) as customer_count,
# MAGIC     AVG(annual_income) as avg_income,
# MAGIC     AVG(age) as avg_age
# MAGIC   FROM customers
# MAGIC   GROUP BY segment
# MAGIC ),
# MAGIC segment_ranking AS (
# MAGIC   SELECT 
# MAGIC     segment,
# MAGIC     customer_count,
# MAGIC     avg_income,
# MAGIC     avg_age,
# MAGIC     RANK() OVER (ORDER BY customer_count DESC) as size_rank,
# MAGIC     RANK() OVER (ORDER BY avg_income DESC) as income_rank
# MAGIC   FROM customer_stats
# MAGIC )
# MAGIC SELECT 
# MAGIC   segment,
# MAGIC   customer_count,
# MAGIC   ROUND(avg_income, 2) as avg_income,
# MAGIC   ROUND(avg_age, 1) as avg_age,
# MAGIC   size_rank,
# MAGIC   income_rank
# MAGIC FROM segment_ranking
# MAGIC ORDER BY size_rank

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Complex CTE: Product performance analysis
# MAGIC WITH product_metrics AS (
# MAGIC   SELECT 
# MAGIC     category,
# MAGIC     brand,
# MAGIC     COUNT(*) as product_count,
# MAGIC     AVG(price) as avg_price,
# MAGIC     AVG(rating) as avg_rating,
# MAGIC     SUM(num_reviews) as total_reviews
# MAGIC   FROM products
# MAGIC   WHERE NOT discontinued
# MAGIC   GROUP BY category, brand
# MAGIC ),
# MAGIC category_summary AS (
# MAGIC   SELECT 
# MAGIC     category,
# MAGIC     AVG(avg_price) as category_avg_price,
# MAGIC     AVG(avg_rating) as category_avg_rating
# MAGIC   FROM product_metrics
# MAGIC   GROUP BY category
# MAGIC )
# MAGIC SELECT 
# MAGIC   pm.category,
# MAGIC   pm.brand,
# MAGIC   pm.product_count,
# MAGIC   ROUND(pm.avg_price, 2) as brand_avg_price,
# MAGIC   ROUND(cs.category_avg_price, 2) as category_avg_price,
# MAGIC   ROUND(pm.avg_rating, 2) as brand_rating,
# MAGIC   ROUND(cs.category_avg_rating, 2) as category_rating,
# MAGIC   pm.total_reviews
# MAGIC FROM product_metrics pm
# MAGIC JOIN category_summary cs ON pm.category = cs.category
# MAGIC ORDER BY pm.category, pm.product_count DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Subqueries
# MAGIC
# MAGIC **Types:**
# MAGIC - Scalar subquery: Returns single value
# MAGIC - Row subquery: Returns single row
# MAGIC - Table subquery: Returns multiple rows/columns

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Subquery: Find customers with above-average income
# MAGIC SELECT 
# MAGIC   customer_id,
# MAGIC   first_name,
# MAGIC   last_name,
# MAGIC   annual_income,
# MAGIC   segment
# MAGIC FROM customers
# MAGIC WHERE annual_income > (SELECT AVG(annual_income) FROM customers)
# MAGIC ORDER BY annual_income DESC
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Correlated subquery: Find top products by category
# MAGIC SELECT 
# MAGIC   p1.category,
# MAGIC   p1.product_name,
# MAGIC   p1.price,
# MAGIC   p1.rating
# MAGIC FROM products p1
# MAGIC WHERE p1.rating >= (
# MAGIC   SELECT AVG(p2.rating)
# MAGIC   FROM products p2
# MAGIC   WHERE p2.category = p1.category
# MAGIC )
# MAGIC ORDER BY p1.category, p1.rating DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data Warehousing Concepts
# MAGIC
# MAGIC ### Creating Tables
# MAGIC
# MAGIC **Table Types in Databricks:**
# MAGIC - **Managed Tables**: Databricks manages both metadata and data
# MAGIC - **External Tables**: Only metadata managed, data stored externally
# MAGIC - **Delta Tables**: Optimized format with ACID transactions

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create a managed table from query results
# MAGIC CREATE OR REPLACE TABLE customer_segments AS
# MAGIC SELECT 
# MAGIC   segment,
# MAGIC   COUNT(*) as customer_count,
# MAGIC   AVG(age) as avg_age,
# MAGIC   AVG(annual_income) as avg_income,
# MAGIC   MIN(annual_income) as min_income,
# MAGIC   MAX(annual_income) as max_income,
# MAGIC   STDDEV(annual_income) as stddev_income
# MAGIC FROM customers
# MAGIC GROUP BY segment

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Query the newly created table
# MAGIC SELECT * FROM customer_segments
# MAGIC ORDER BY customer_count DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create table with explicit schema
# MAGIC CREATE OR REPLACE TABLE sales_summary (
# MAGIC   region STRING,
# MAGIC   total_transactions INT,
# MAGIC   total_revenue DOUBLE,
# MAGIC   avg_satisfaction DOUBLE,
# MAGIC   total_quantity INT
# MAGIC )
# MAGIC USING DELTA;
# MAGIC
# MAGIC -- Insert data into the table
# MAGIC INSERT INTO sales_summary
# MAGIC SELECT 
# MAGIC   region,
# MAGIC   COUNT(transaction_id) as total_transactions,
# MAGIC   SUM(total_sales) as total_revenue,
# MAGIC   AVG(customer_satisfaction) as avg_satisfaction,
# MAGIC   SUM(quantity) as total_quantity
# MAGIC FROM sales
# MAGIC GROUP BY region

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM sales_summary
# MAGIC ORDER BY total_revenue DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating Views
# MAGIC
# MAGIC **Views vs Tables:**
# MAGIC - Views are virtual tables (don't store data)
# MAGIC - Always reflect current data from source tables
# MAGIC - Useful for simplifying complex queries
# MAGIC - Can control access to sensitive data

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create a temporary view for high-value customers
# MAGIC CREATE OR REPLACE TEMPORARY VIEW high_value_customers AS
# MAGIC SELECT 
# MAGIC   customer_id,
# MAGIC   first_name,
# MAGIC   last_name,
# MAGIC   email,
# MAGIC   city,
# MAGIC   state,
# MAGIC   segment,
# MAGIC   annual_income,
# MAGIC   age
# MAGIC FROM customers
# MAGIC WHERE annual_income > 75000
# MAGIC   AND email_subscribed = true

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Query the view
# MAGIC SELECT segment, COUNT(*) as count, AVG(annual_income) as avg_income
# MAGIC FROM high_value_customers
# MAGIC GROUP BY segment
# MAGIC ORDER BY count DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW product_performance AS
# MAGIC SELECT 
# MAGIC   category,
# MAGIC   product_name,
# MAGIC   brand,
# MAGIC   price,
# MAGIC   rating,
# MAGIC   num_reviews,
# MAGIC   price * num_reviews as revenue_potential,
# MAGIC   CASE 
# MAGIC     WHEN rating >= 4.5 THEN 'Excellent'
# MAGIC     WHEN rating >= 4.0 THEN 'Good'
# MAGIC     WHEN rating >= 3.0 THEN 'Average'
# MAGIC     ELSE 'Poor'
# MAGIC   END as rating_category,
# MAGIC   CASE
# MAGIC     WHEN price < 50 THEN 'Budget'
# MAGIC     WHEN price < 100 THEN 'Mid-Range'
# MAGIC     ELSE 'Premium'
# MAGIC   END as price_tier
# MAGIC FROM products
# MAGIC WHERE NOT discontinued

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Analyze using the view
# MAGIC SELECT 
# MAGIC   category,
# MAGIC   rating_category,
# MAGIC   COUNT(*) as product_count,
# MAGIC   AVG(price) as avg_price
# MAGIC FROM product_performance
# MAGIC GROUP BY category, rating_category
# MAGIC ORDER BY category, rating_category

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query Optimization
# MAGIC
# MAGIC **Optimization Techniques:**
# MAGIC 1. **Partitioning**: Organize data by column values
# MAGIC 2. **Caching**: Store frequently accessed data in memory
# MAGIC 3. **Predicate Pushdown**: Filter data early
# MAGIC 4. **Column Pruning**: Select only needed columns
# MAGIC 5. **Broadcast Joins**: For small tables

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Use EXPLAIN to see query execution plan
# MAGIC EXPLAIN FORMATTED
# MAGIC SELECT 
# MAGIC   c.segment,
# MAGIC   COUNT(*) as customer_count,
# MAGIC   AVG(c.annual_income) as avg_income
# MAGIC FROM customers c
# MAGIC WHERE c.state IN ('California', 'Texas', 'New York')
# MAGIC GROUP BY c.segment

# COMMAND ----------

# Remove caching, just trigger an action to materialize the DataFrame
count = customer_df.count()
print(f"Processed {count} customer records")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Advanced Aggregations
# MAGIC
# MAGIC **CUBE and ROLLUP:**
# MAGIC - `ROLLUP`: Creates subtotals and grand totals (hierarchical)
# MAGIC - `CUBE`: Creates all possible aggregation combinations
# MAGIC - `GROUPING SETS`: Specify exact grouping combinations

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ROLLUP: Hierarchical aggregation
# MAGIC SELECT 
# MAGIC   state,
# MAGIC   segment,
# MAGIC   COUNT(*) as customer_count,
# MAGIC   AVG(annual_income) as avg_income
# MAGIC FROM customers
# MAGIC WHERE state IN ('California', 'Texas', 'New York', 'Florida')
# MAGIC GROUP BY ROLLUP (state, segment)
# MAGIC ORDER BY state, segment

# COMMAND ----------

# MAGIC %sql
# MAGIC -- CUBE: All combinations
# MAGIC SELECT 
# MAGIC   segment,
# MAGIC   CASE 
# MAGIC     WHEN age < 30 THEN 'Young'
# MAGIC     WHEN age < 50 THEN 'Middle'
# MAGIC     ELSE 'Senior'
# MAGIC   END as age_group,
# MAGIC   COUNT(*) as customer_count,
# MAGIC   AVG(annual_income) as avg_income
# MAGIC FROM customers
# MAGIC GROUP BY CUBE (segment, age_group)
# MAGIC ORDER BY segment, age_group

# COMMAND ----------

# MAGIC %md
# MAGIC ### PIVOT Operations
# MAGIC
# MAGIC **Purpose:** Transform rows to columns

# COMMAND ----------

# MAGIC %sql
# MAGIC -- PIVOT: Customer segments by state
# MAGIC SELECT * FROM (
# MAGIC   SELECT state, segment, customer_id
# MAGIC   FROM customers
# MAGIC   WHERE state IN ('California', 'Texas', 'New York', 'Florida', 'Illinois')
# MAGIC )
# MAGIC PIVOT (
# MAGIC   COUNT(customer_id)
# MAGIC   FOR segment IN ('Premium', 'Standard', 'Basic', 'VIP')
# MAGIC )
# MAGIC ORDER BY state

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Performance Best Practices
# MAGIC
# MAGIC ### Tips for Writing Efficient Queries:
# MAGIC
# MAGIC 1. **Filter Early**: Use WHERE clauses to reduce data volume
# MAGIC 2. **Select Only Needed Columns**: Avoid SELECT *
# MAGIC 3. **Use Appropriate Joins**: Choose the right join type
# MAGIC 4. **Leverage Partitioning**: Query on partitioned columns
# MAGIC 5. **Cache Frequently Used Data**: Use .cache() for repeated access
# MAGIC 6. **Use Delta Tables**: Better performance and features
# MAGIC 7. **Optimize Window Functions**: Limit partition size when possible
# MAGIC 8. **Avoid Cartesian Joins**: Always include join conditions

# COMMAND ----------

# Example of optimized query structure

# BAD: Select everything, filter late
# SELECT * FROM large_table WHERE condition

# GOOD: Select specific columns, filter early
# SELECT col1, col2, col3 FROM large_table WHERE condition

# Demonstrate with explain
spark.sql("""
  SELECT customer_id, first_name, last_name, annual_income
  FROM customers
  WHERE state = 'California' 
    AND annual_income > 50000
  LIMIT 100
""").explain()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Takeaways
# MAGIC
# MAGIC **SQL Fundamentals:**
# MAGIC - Use temporary views to query DataFrames with SQL
# MAGIC - Master SELECT, WHERE, GROUP BY, and JOIN operations
# MAGIC - Understand aggregate functions and their use cases
# MAGIC
# MAGIC **Advanced SQL:**
# MAGIC - Window functions enable powerful row-based calculations
# MAGIC - CTEs improve query readability and maintainability
# MAGIC - Subqueries provide flexible data filtering
# MAGIC
# MAGIC **Data Warehousing:**
# MAGIC - Create tables and views for persistent storage
# MAGIC - Use Delta Lake format for ACID transactions
# MAGIC - Optimize queries with caching, partitioning, and proper indexing
# MAGIC
# MAGIC **Performance:**
# MAGIC - Filter early, select specific columns
# MAGIC - Use EXPLAIN to understand query plans
# MAGIC - Cache frequently accessed data
# MAGIC
# MAGIC **Next Steps:** Move to Notebook 4 for ETL & Data Processing!
