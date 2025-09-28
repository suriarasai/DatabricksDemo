# Quick Start Guide - Databricks Demo

## 🚀 5-Minute Quick Start

### Step 1: Sign up for Databricks Community Edition
1. Go to [community.cloud.databricks.com](https://community.cloud.databricks.com/)
2. Create your free account
3. Verify your email

### Step 2: Create a Cluster
1. Go to **Compute** → **Create Cluster**
2. Name: `demo-cluster`
3. Runtime: Latest LTS version
4. Click **Create Cluster** (takes 3-5 minutes)

### Step 3: Import Your First Notebook
1. Download: `notebooks/01-basics/spark-fundamentals.py`
2. In Databricks: **Workspace** → **Import**
3. Upload the file

### Step 4: Run Your First Analysis
1. Attach notebook to your cluster
2. Run the first few cells
3. Observe the results!

## 📊 What You'll Learn

| Notebook | Skills | Duration |
|----------|--------|----------|
| **01-basics/spark-fundamentals** | DataFrames, SQL, basic operations | 30 min |
| **02-etl/ecommerce-pipeline** | Data processing, ETL patterns | 45 min |
| **03-analytics/visualization-dashboard** | Charts, dashboards, BI | 30 min |
| **04-ml/customer-churn-prediction** | Machine learning, MLlib | 60 min |
| **05-streaming/iot-sensor-streaming** | Real-time processing | 45 min |

## 💡 Pro Tips

### For Beginners
- Start with built-in datasets: `/databricks-datasets/`
- Use `display()` instead of `show()` for better output
- Save your work frequently (auto-save is enabled)

### For Intermediate Users
- Cache frequently used DataFrames: `df.cache()`
- Use Parquet format for better performance
- Leverage SQL interface for complex queries

### For Advanced Users
- Explore Delta Lake for ACID transactions
- Try MLflow for ML lifecycle management
- Use Structured Streaming for real-time processing

## 🆘 Common Issues

| Problem | Solution |
|---------|----------|
| Cluster won't start | Wait 5 minutes, try different runtime |
| Out of memory | Use smaller datasets or increase partitions |
| Import errors | Restart cluster, check library versions |

## 📚 Next Steps

1. **Complete all 5 notebooks** in order
2. **Experiment with your own data**
3. **Build a personal project**
4. **Share your learnings** with the community

## 🌟 Success Metrics

By the end of this demo, you should be able to:
- ✅ Create and manipulate Spark DataFrames
- ✅ Write efficient SQL queries for analytics
- ✅ Build basic ETL pipelines
- ✅ Create interactive visualizations
- ✅ Train and evaluate ML models
- ✅ Process streaming data in real-time

## 🔗 Helpful Resources

- **Documentation**: [docs.databricks.com](https://docs.databricks.com/)
- **Community**: [community.databricks.com](https://community.databricks.com/)
- **Training**: [academy.databricks.com](https://academy.databricks.com/)
- **Certification**: [databricks.com/learn/certification](https://databricks.com/learn/certification)

Happy learning! 🎉