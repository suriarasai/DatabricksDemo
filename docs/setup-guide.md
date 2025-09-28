# Databricks Community Edition Setup Guide

This guide will help you get started with Databricks Community Edition and set up this demo repository.

## 🚀 Quick Start

### Step 1: Sign up for Databricks Community Edition

1. Go to [community.cloud.databricks.com](https://community.cloud.databricks.com/)
2. Click "Sign up for Community Edition"
3. Fill out the registration form with your email
4. Verify your email address
5. Complete your profile setup

**Important**: Choose "Community Edition" (not the paid trial) to access the free tier.

### Step 2: Access Your Workspace

1. Log in to your Databricks workspace
2. You'll see the main Databricks interface with:
   - **Workspace**: For organizing notebooks and folders
   - **Data**: For managing data sources and tables
   - **Compute**: For creating and managing clusters
   - **Workflows**: For scheduling jobs (limited in Community Edition)

### Step 3: Create Your First Cluster

1. Navigate to **Compute** in the sidebar
2. Click **Create Cluster**
3. Configure your cluster:
   - **Cluster Name**: `demo-cluster` or any name you prefer
   - **Cluster Mode**: `Single Node` (only option in Community Edition)
   - **Databricks Runtime Version**: Select the latest LTS version
   - **Node Type**: `Single Node` with up to 15GB memory
4. Click **Create Cluster**
5. Wait 3-5 minutes for the cluster to start

## 📚 Importing the Demo Notebooks

### Method 1: Import Individual Notebooks

1. In your Databricks workspace, go to **Workspace**
2. Right-click on your user folder
3. Select **Import**
4. Choose **File** and upload notebook files from this repository
5. Repeat for each notebook you want to import

### Method 2: Clone from Git (Recommended)

1. In the **Workspace**, right-click on your user folder
2. Select **Create > Folder** and name it `DatabricksDemo`
3. Inside the folder, right-click and select **Import**
4. Choose **URL** and enter: `https://github.com/suriarasai/DatabricksDemo.git`
5. Click **Import**

## 🎯 Running Your First Notebook

### Start with the Basics

1. Open `notebooks/01-basics/spark-fundamentals.py`
2. Attach it to your running cluster (dropdown at the top)
3. Run each cell by pressing `Shift + Enter`
4. Observe the output and explore the explanations

### Key Things to Remember

- **Always attach notebooks to a running cluster**
- **Use `display()` instead of `show()` for better visualization**
- **Built-in datasets** are available at `/databricks-datasets/`
- **Temporary storage** is available at `/tmp/`

## 🛠️ Databricks Community Edition Features

### ✅ What's Included (Free)

| Feature | Community Edition | Notes |
|---------|------------------|--------|
| **Cluster** | Single node (up to 15GB RAM) | Perfect for learning and small datasets |
| **Storage** | 15GB workspace storage | Includes notebooks, libraries, data |
| **Compute Time** | No time limits | Clusters auto-terminate after 2 hours of inactivity |
| **Languages** | Python, Scala, R, SQL | Full language support |
| **Libraries** | Pre-installed ML libraries | NumPy, Pandas, Scikit-learn, MLlib, etc. |
| **Data Sources** | File uploads, cloud storage | CSV, JSON, Parquet, Delta Lake |
| **Visualization** | Built-in charts and dashboards | Interactive plotting capabilities |
| **Collaboration** | Notebook sharing | Share read-only notebooks |

### ❌ Limitations

| Feature | Limitation | Workaround |
|---------|------------|------------|
| **Multi-node clusters** | Not available | Use single-node optimized code |
| **Jobs/Scheduling** | Limited | Run notebooks manually |
| **Advanced security** | Basic only | Suitable for learning/development |
| **Production features** | Not included | Upgrade to paid plan for production |

## 📊 Working with Data

### Built-in Sample Datasets

Databricks provides sample datasets perfect for learning:

```python
# List available datasets
dbutils.fs.ls("/databricks-datasets/")

# Common datasets for practice
diamonds = spark.read.csv("/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv", header=True, inferSchema=True)
airlines = spark.read.csv("/databricks-datasets/airlines/", header=True, inferSchema=True)
```

### Uploading Your Own Data

1. Go to **Data** in the sidebar
2. Click **Create Table**
3. Choose **Upload File**
4. Drag and drop your file (CSV, JSON, etc.)
5. Follow the wizard to create a table

### Cloud Storage Integration

```python
# AWS S3 (example)
df = spark.read.csv("s3a://your-bucket/path/to/file.csv")

# Azure Blob Storage (example)
df = spark.read.csv("abfss://container@account.dfs.core.windows.net/path/file.csv")

# Google Cloud Storage (example)
df = spark.read.csv("gs://your-bucket/path/to/file.csv")
```

## 🔧 Essential Databricks Commands

### File System Commands (dbutils)

```python
# List files and directories
dbutils.fs.ls("/")
dbutils.fs.ls("/databricks-datasets/")

# Copy files
dbutils.fs.cp("source_path", "destination_path")

# Remove files/directories
dbutils.fs.rm("/tmp/my_temp_data", recurse=True)

# Create directories
dbutils.fs.mkdirs("/tmp/my_new_directory")
```

### Display and Visualization

```python
# Display DataFrames (better than .show())
display(df)

# Display with custom settings
display(df.limit(100))

# Create visualizations
# After running display(), click the chart icon to create plots
```

### Spark Configuration

```python
# Check Spark version and config
print(spark.version)
print(spark.conf.get("spark.master"))

# Set configuration (if needed)
spark.conf.set("spark.sql.adaptive.enabled", "true")
```

## 🎓 Learning Path

### 🌟 Beginner (Week 1-2)
1. **Spark Fundamentals** - `01-basics/spark-fundamentals.py`
   - DataFrames, basic operations, SQL interface
   - Focus: Understanding Spark concepts and API

2. **Data Exploration** - Work with built-in datasets
   - Practice filtering, grouping, aggregations
   - Focus: Getting comfortable with data manipulation

### 🚀 Intermediate (Week 3-4)
3. **ETL Pipeline** - `02-etl/ecommerce-pipeline.py`
   - Data ingestion, transformation, loading
   - Focus: Building end-to-end data pipelines

4. **Analytics and Visualization** - `03-analytics/` (when available)
   - Advanced SQL, window functions, statistical analysis
   - Focus: Deriving insights from data

### 🎯 Advanced (Week 5-6)
5. **Machine Learning** - `04-ml/customer-churn-prediction.py`
   - MLlib, feature engineering, model evaluation
   - Focus: Building predictive models

6. **Streaming** - `05-streaming/iot-sensor-streaming.py`
   - Real-time data processing, structured streaming
   - Focus: Handling continuous data streams

## 💡 Tips for Success

### Best Practices

1. **Start Small**: Begin with small datasets to understand concepts
2. **Use display()**: Better visualization than show() in Databricks
3. **Leverage Built-in Datasets**: Great for learning without setup
4. **Save Frequently**: Notebooks auto-save, but create checkpoints
5. **Monitor Cluster**: Watch resource usage and restart if needed

### Common Gotchas

1. **Cluster Auto-termination**: Clusters stop after 2 hours of inactivity
2. **Memory Limits**: Single node has 15GB limit - be mindful of data size
3. **File Paths**: Use `/tmp/` for temporary storage
4. **Case Sensitivity**: SQL and column names are case-sensitive
5. **Lazy Evaluation**: Transformations don't execute until an action is called

### Performance Tips

```python
# Cache frequently used DataFrames
df_cached = df.cache()
df_cached.count()  # Triggers caching

# Use appropriate file formats
df.write.parquet("/tmp/my_data.parquet")  # Efficient columnar format

# Partition large datasets
df.write.partitionBy("year").parquet("/tmp/partitioned_data")

# Use broadcast for small lookup tables
from pyspark.sql.functions import broadcast
result = large_df.join(broadcast(small_df), "key")
```

## 🆘 Troubleshooting

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| **Cluster won't start** | Wait 5-10 minutes, try different runtime version |
| **Out of memory errors** | Reduce data size, increase partitions, clear cache |
| **Slow performance** | Cache DataFrames, use efficient file formats |
| **Import errors** | Restart cluster, check library availability |
| **File not found** | Verify file paths, check file system with `dbutils.fs.ls()` |

### Getting Help

1. **Databricks Documentation**: [docs.databricks.com](https://docs.databricks.com/)
2. **Community Forums**: [community.databricks.com](https://community.databricks.com/)
3. **Stack Overflow**: Tag questions with `databricks` and `apache-spark`
4. **Apache Spark Docs**: [spark.apache.org/docs/latest/](https://spark.apache.org/docs/latest/)

## 🌟 Next Steps

Once you've mastered the basics:

1. **Explore Advanced Features**: Delta Lake, MLflow integration
2. **Build Real Projects**: Use your own data and business problems
3. **Consider Certification**: Databricks offers certification programs
4. **Join the Community**: Participate in forums and user groups
5. **Upgrade When Ready**: Move to paid tiers for production workloads

## 📞 Support

If you encounter issues with this demo repository:

1. Check the troubleshooting section above
2. Review the notebook documentation and comments
3. Search existing issues on GitHub
4. Create a new issue with detailed error information

Happy learning with Databricks! 🚀