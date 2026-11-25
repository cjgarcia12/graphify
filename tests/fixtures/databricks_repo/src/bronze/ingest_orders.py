# Databricks notebook source
# MAGIC %run ../common/config

# COMMAND ----------

raw = spark.read.format("json").load("/Volumes/raw/landing/orders")
raw.write.mode("append").saveAsTable(f"{catalog}.bronze.orders_raw")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS ${catalog}.bronze.customers_raw AS
# MAGIC SELECT * FROM read_files('/Volumes/raw/landing/customers')
