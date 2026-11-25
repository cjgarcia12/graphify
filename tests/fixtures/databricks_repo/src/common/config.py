# Databricks notebook source
dbutils.widgets.text("catalog", "dev")
catalog = dbutils.widgets.get("catalog")

# COMMAND ----------

def bronze_table(name):
    return f"{catalog}.bronze.{name}"
