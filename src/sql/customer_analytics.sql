-- Customer Analytics SQL Examples for Databricks
-- ================================================
-- This file contains common SQL patterns and queries for customer analytics
-- Designed to showcase SQL capabilities in Databricks

-- =============================================
-- Customer Segmentation Analysis
-- =============================================

-- RFM Analysis (Recency, Frequency, Monetary)
WITH customer_rfm AS (
  SELECT 
    customer_id,
    DATEDIFF(CURRENT_DATE(), MAX(order_date)) as recency_days,
    COUNT(DISTINCT order_id) as frequency,
    SUM(order_total) as monetary_value
  FROM orders
  GROUP BY customer_id
),

rfm_scores AS (
  SELECT 
    *,
    NTILE(5) OVER (ORDER BY recency_days DESC) as recency_score,
    NTILE(5) OVER (ORDER BY frequency) as frequency_score,
    NTILE(5) OVER (ORDER BY monetary_value) as monetary_score
  FROM customer_rfm
),

customer_segments AS (
  SELECT 
    *,
    (recency_score + frequency_score + monetary_score) as rfm_total,
    CASE 
      WHEN recency_score >= 4 AND frequency_score >= 4 AND monetary_score >= 4 THEN 'Champions'
      WHEN recency_score >= 3 AND frequency_score >= 3 AND monetary_score >= 3 THEN 'Loyal Customers'
      WHEN recency_score >= 3 AND frequency_score <= 2 THEN 'Potential Loyalists'
      WHEN recency_score <= 2 AND frequency_score >= 3 THEN 'At Risk'
      WHEN recency_score <= 2 AND frequency_score <= 2 THEN 'Lost Customers'
      ELSE 'New Customers'
    END as segment
  FROM rfm_scores
)

SELECT 
  segment,
  COUNT(*) as customer_count,
  AVG(monetary_value) as avg_customer_value,
  AVG(frequency) as avg_order_frequency,
  AVG(recency_days) as avg_days_since_last_order
FROM customer_segments
GROUP BY segment
ORDER BY avg_customer_value DESC;

-- =============================================
-- Customer Lifetime Value (CLV) Analysis
-- =============================================

WITH customer_metrics AS (
  SELECT 
    customer_id,
    MIN(order_date) as first_order_date,
    MAX(order_date) as last_order_date,
    COUNT(DISTINCT order_id) as total_orders,
    SUM(order_total) as total_spent,
    AVG(order_total) as avg_order_value,
    DATEDIFF(MAX(order_date), MIN(order_date)) as customer_lifespan_days
  FROM orders
  GROUP BY customer_id
),

clv_calculation AS (
  SELECT 
    *,
    CASE 
      WHEN customer_lifespan_days > 0 
      THEN (total_spent / customer_lifespan_days) * 365 
      ELSE total_spent 
    END as estimated_annual_value,
    
    CASE 
      WHEN customer_lifespan_days > 0 
      THEN total_orders / (customer_lifespan_days / 30.0) 
      ELSE total_orders 
    END as orders_per_month
  FROM customer_metrics
)

SELECT 
  customer_id,
  total_spent,
  avg_order_value,
  orders_per_month,
  estimated_annual_value,
  CASE 
    WHEN estimated_annual_value >= 1000 THEN 'High Value'
    WHEN estimated_annual_value >= 500 THEN 'Medium Value'
    ELSE 'Low Value'
  END as value_segment
FROM clv_calculation
ORDER BY estimated_annual_value DESC;

-- =============================================
-- Cohort Analysis
-- =============================================

WITH customer_cohorts AS (
  SELECT 
    customer_id,
    DATE_TRUNC('month', MIN(order_date)) as cohort_month,
    DATE_TRUNC('month', order_date) as order_month
  FROM orders
  GROUP BY customer_id, DATE_TRUNC('month', order_date)
),

cohort_data AS (
  SELECT 
    cohort_month,
    order_month,
    COUNT(DISTINCT customer_id) as customers,
    MONTHS_BETWEEN(order_month, cohort_month) as period_number
  FROM customer_cohorts
  GROUP BY cohort_month, order_month
),

cohort_sizes AS (
  SELECT 
    cohort_month,
    COUNT(DISTINCT customer_id) as cohort_size
  FROM customer_cohorts
  WHERE cohort_month = order_month
  GROUP BY cohort_month
)

SELECT 
  cd.cohort_month,
  cd.period_number,
  cd.customers,
  cs.cohort_size,
  ROUND(cd.customers * 100.0 / cs.cohort_size, 2) as retention_rate
FROM cohort_data cd
JOIN cohort_sizes cs ON cd.cohort_month = cs.cohort_month
ORDER BY cd.cohort_month, cd.period_number;

-- =============================================
-- Product Affinity Analysis
-- =============================================

WITH product_pairs AS (
  SELECT 
    oi1.product_id as product_a,
    oi2.product_id as product_b,
    COUNT(DISTINCT oi1.order_id) as co_occurrence_count
  FROM order_items oi1
  JOIN order_items oi2 ON oi1.order_id = oi2.order_id
  WHERE oi1.product_id < oi2.product_id  -- Avoid duplicates and self-pairs
  GROUP BY oi1.product_id, oi2.product_id
),

product_totals AS (
  SELECT 
    product_id,
    COUNT(DISTINCT order_id) as total_orders
  FROM order_items
  GROUP BY product_id
)

SELECT 
  pp.product_a,
  p1.product_name as product_a_name,
  pp.product_b,
  p2.product_name as product_b_name,
  pp.co_occurrence_count,
  pt1.total_orders as product_a_total_orders,
  pt2.total_orders as product_b_total_orders,
  ROUND(pp.co_occurrence_count * 100.0 / pt1.total_orders, 2) as lift_a_to_b,
  ROUND(pp.co_occurrence_count * 100.0 / pt2.total_orders, 2) as lift_b_to_a
FROM product_pairs pp
JOIN products p1 ON pp.product_a = p1.product_id
JOIN products p2 ON pp.product_b = p2.product_id
JOIN product_totals pt1 ON pp.product_a = pt1.product_id
JOIN product_totals pt2 ON pp.product_b = pt2.product_id
WHERE pp.co_occurrence_count >= 10  -- Minimum threshold for significance
ORDER BY pp.co_occurrence_count DESC;

-- =============================================
-- Customer Churn Risk Analysis
-- =============================================

WITH customer_activity AS (
  SELECT 
    customer_id,
    MAX(order_date) as last_order_date,
    COUNT(DISTINCT order_id) as total_orders,
    AVG(DATEDIFF(order_date, LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date))) as avg_days_between_orders,
    SUM(order_total) as total_spent
  FROM orders
  GROUP BY customer_id
),

churn_risk AS (
  SELECT 
    *,
    DATEDIFF(CURRENT_DATE(), last_order_date) as days_since_last_order,
    CASE 
      WHEN avg_days_between_orders IS NULL THEN 30  -- Single order customers
      ELSE avg_days_between_orders 
    END as expected_days_between_orders
  FROM customer_activity
)

SELECT 
  customer_id,
  days_since_last_order,
  expected_days_between_orders,
  total_orders,
  total_spent,
  CASE 
    WHEN days_since_last_order > (expected_days_between_orders * 2) THEN 'High Risk'
    WHEN days_since_last_order > expected_days_between_orders THEN 'Medium Risk'
    ELSE 'Low Risk'
  END as churn_risk_level,
  ROUND(days_since_last_order / expected_days_between_orders, 2) as risk_multiplier
FROM churn_risk
ORDER BY risk_multiplier DESC;

-- =============================================
-- Sales Performance Dashboard Queries
-- =============================================

-- Daily Sales Summary
SELECT 
  DATE(order_date) as sale_date,
  COUNT(DISTINCT order_id) as total_orders,
  COUNT(DISTINCT customer_id) as unique_customers,
  SUM(order_total) as total_revenue,
  AVG(order_total) as avg_order_value,
  SUM(order_total) / COUNT(DISTINCT customer_id) as revenue_per_customer
FROM orders
WHERE order_date >= CURRENT_DATE() - INTERVAL 30 DAYS
GROUP BY DATE(order_date)
ORDER BY sale_date DESC;

-- Monthly Sales Trends
SELECT 
  DATE_TRUNC('month', order_date) as month,
  SUM(order_total) as monthly_revenue,
  COUNT(DISTINCT order_id) as monthly_orders,
  COUNT(DISTINCT customer_id) as monthly_customers,
  LAG(SUM(order_total)) OVER (ORDER BY DATE_TRUNC('month', order_date)) as prev_month_revenue,
  ROUND(
    (SUM(order_total) - LAG(SUM(order_total)) OVER (ORDER BY DATE_TRUNC('month', order_date))) * 100.0 / 
    LAG(SUM(order_total)) OVER (ORDER BY DATE_TRUNC('month', order_date)), 2
  ) as revenue_growth_rate
FROM orders
GROUP BY DATE_TRUNC('month', order_date)
ORDER BY month DESC;

-- Product Performance Analysis
SELECT 
  p.category,
  p.product_name,
  COUNT(DISTINCT oi.order_process
  SUM(oi.quantity) as units_sold,
  SUM(oi.quantity * oi.unit_price) as total_revenue,
  AVG(oi.unit_price) as avg_selling_price,
  COUNT(DISTINCT o.customer_id) as unique_customers
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
JOIN products p ON oi.product_id = p.product_id
WHERE o.order_date >= CURRENT_DATE() - INTERVAL 90 DAYS
GROUP BY p.category, p.product_name
ORDER BY total_revenue DESC;

-- =============================================
-- Advanced Window Function Examples
-- =============================================

-- Running totals and moving averages
WITH daily_sales AS (
  SELECT 
    DATE(order_date) as sale_date,
    SUM(order_total) as daily_revenue
  FROM orders
  GROUP BY DATE(order_date)
)

SELECT 
  sale_date,
  daily_revenue,
  SUM(daily_revenue) OVER (ORDER BY sale_date) as running_total,
  AVG(daily_revenue) OVER (
    ORDER BY sale_date 
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) as moving_avg_7_days,
  RANK() OVER (ORDER BY daily_revenue DESC) as revenue_rank,
  LAG(daily_revenue, 1) OVER (ORDER BY sale_date) as prev_day_revenue,
  ROUND(
    (daily_revenue - LAG(daily_revenue, 1) OVER (ORDER BY sale_date)) * 100.0 / 
    LAG(daily_revenue, 1) OVER (ORDER BY sale_date), 2
  ) as day_over_day_growth
FROM daily_sales
ORDER BY sale_date DESC;

-- Customer ranking within segments
WITH customer_segments AS (
  SELECT 
    customer_id,
    SUM(order_total) as total_spent,
    COUNT(DISTINCT order_id) as total_orders,
    CASE 
      WHEN SUM(order_total) >= 1000 THEN 'Premium'
      WHEN SUM(order_total) >= 500 THEN 'Standard'
      ELSE 'Basic'
    END as customer_tier
  FROM orders
  GROUP BY customer_id
)

SELECT 
  customer_id,
  customer_tier,
  total_spent,
  total_orders,
  RANK() OVER (PARTITION BY customer_tier ORDER BY total_spent DESC) as tier_rank,
  PERCENT_RANK() OVER (PARTITION BY customer_tier ORDER BY total_spent) as percentile_within_tier,
  NTILE(10) OVER (PARTITION BY customer_tier ORDER BY total_spent) as decile_within_tier
FROM customer_segments
ORDER BY customer_tier, tier_rank;