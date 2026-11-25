USE CATALOG ${catalog};
CREATE OR REPLACE VIEW reporting.revenue_by_month AS
SELECT date_trunc('month', order_date) AS month, sum(amount) AS revenue
FROM gold.daily_revenue
GROUP BY 1;
