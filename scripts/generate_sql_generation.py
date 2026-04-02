#!/usr/bin/env python3
"""
Generate NL-to-SQL dataset for RosettaStone benchmarks.

Produces 300 NL/SQL pairs across 6 complexity variants and 3 schema domains.
Questions are authored to work with the custom e-commerce, HR, and SaaS schemas.
Spider 1.0 (CC-BY-SA-4.0) is used as inspiration for question patterns in the
simple/join/aggregation/cte_subquery variants.

Calls GPT-4o and Haiku for model responses. Outputs JSONL files.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import sqlparse
from dotenv import load_dotenv
from litellm import completion

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENV_PATH = Path("/Users/ashwinchidambaram/dev/projects/rosettastone/.env")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "examples" / "datasets" / "sql_generation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_PAIRS = 300
DISTRIBUTION = {
    "simple": 60,
    "join": 70,
    "aggregation": 60,
    "cte_subquery": 50,
    "window_function": 30,
    "unanswerable": 30,
}

# ---------------------------------------------------------------------------
# Schema definitions (three domains)
# ---------------------------------------------------------------------------
ECOMMERCE_SCHEMA = """\
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    total_amount NUMERIC(10,2) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('pending','confirmed','shipped','delivered','cancelled')),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    category VARCHAR(100),
    price NUMERIC(10,2) NOT NULL,
    stock_quantity INTEGER DEFAULT 0
);

CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price NUMERIC(10,2) NOT NULL
);"""

HR_SCHEMA = """\
CREATE TABLE departments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    budget NUMERIC(12,2),
    location VARCHAR(100)
);

CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    department_id INTEGER REFERENCES departments(id),
    salary NUMERIC(10,2) NOT NULL,
    hire_date DATE NOT NULL,
    manager_id INTEGER REFERENCES employees(id)
);

CREATE TABLE performance_reviews (
    id SERIAL PRIMARY KEY,
    employee_id INTEGER REFERENCES employees(id),
    review_date DATE NOT NULL,
    score INTEGER CHECK (score BETWEEN 1 AND 5),
    reviewer_id INTEGER REFERENCES employees(id)
);"""

SAAS_SCHEMA = """\
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    plan VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    last_active TIMESTAMP
);

CREATE TABLE subscriptions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    plan_name VARCHAR(50) NOT NULL,
    mrr NUMERIC(10,2) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('active','paused','cancelled')),
    started_at TIMESTAMP DEFAULT NOW(),
    cancelled_at TIMESTAMP
);

CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    event_type VARCHAR(100) NOT NULL,
    properties JSONB,
    occurred_at TIMESTAMP DEFAULT NOW()
);"""

SCHEMAS = {
    "ecommerce": ECOMMERCE_SCHEMA,
    "hr": HR_SCHEMA,
    "saas": SAAS_SCHEMA,
}

# ---------------------------------------------------------------------------
# Prompt template for model calls
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a PostgreSQL expert. Given a database schema and a natural language question, \
generate ONLY a valid PostgreSQL query that answers the question.

Rules:
- Output ONLY the SQL query, no explanations, no markdown fences, no comments, no language tags.
- Use explicit JOIN syntax (INNER JOIN, LEFT JOIN, etc.), never implicit comma joins.
- Use PostgreSQL syntax: COALESCE (not IFNULL), STRING_AGG (not GROUP_CONCAT), \
EXTRACT or TO_CHAR for date parts, NOW() for current timestamp.
- If the question CANNOT be answered from the given schema (e.g., references columns or \
tables that don't exist), respond with ONLY this JSON (no markdown, no prefix):
{"error": "Cannot answer: <specific reason>"}
- Do not invent or assume columns/tables not present in the schema.
- Use single quotes for string literals, not double quotes.
- For aggregations, always include GROUP BY for non-aggregated columns."""

SYSTEM_PROMPT_WINDOW = """\
You are a PostgreSQL expert. Given a database schema and a natural language question, \
generate ONLY a valid PostgreSQL query that answers the question.

Rules:
- Output ONLY the SQL query, no explanations, no markdown fences, no comments, no language tags.
- Use explicit JOIN syntax (INNER JOIN, LEFT JOIN, etc.), never implicit comma joins.
- Use PostgreSQL window functions (RANK, DENSE_RANK, ROW_NUMBER, LAG, LEAD, \
SUM/AVG/MAX/MIN OVER, PERCENT_RANK, etc.) when the question asks for rankings, \
running totals, comparisons with previous/next rows, or per-partition aggregates.
- The schema IS sufficient to answer these questions. Do NOT return error JSON.
- Use single quotes for string literals, not double quotes.
- For aggregations, always include GROUP BY for non-aggregated columns."""


def build_prompt(schema_ddl: str, question: str) -> str:
    """Build the user prompt with schema context and question."""
    return f"""Given the following PostgreSQL schema:

{schema_ddl}

Write a PostgreSQL query to answer this question:
{question}"""


# ---------------------------------------------------------------------------
# All 300 question/domain pairs organized by variant
# ---------------------------------------------------------------------------

# SIMPLE (60): Single-table SELECT with WHERE/ORDER BY
# Inspired by Spider "easy" difficulty patterns
SIMPLE_PAIRS = [
    # E-commerce (20)
    {"domain": "ecommerce", "question": "List all customers ordered by name alphabetically."},
    {"domain": "ecommerce", "question": "Show all products in the 'Electronics' category."},
    {"domain": "ecommerce", "question": "Find all orders with status 'delivered'."},
    {
        "domain": "ecommerce",
        "question": "What are the names and prices of products that cost more than 100?",
    },
    {"domain": "ecommerce", "question": "Show all customers who signed up after January 1, 2024."},
    {"domain": "ecommerce", "question": "List the 10 most expensive products."},
    {"domain": "ecommerce", "question": "Find all orders with a total amount between 50 and 200."},
    {"domain": "ecommerce", "question": "Show all products with zero stock."},
    {"domain": "ecommerce", "question": "List all distinct product categories."},
    {"domain": "ecommerce", "question": "Find the product with the highest price."},
    {
        "domain": "ecommerce",
        "question": "Show all cancelled orders sorted by creation date descending.",
    },
    {"domain": "ecommerce", "question": "List all customers whose email contains 'gmail.com'."},
    {"domain": "ecommerce", "question": "What are the names of products priced under 25 dollars?"},
    {"domain": "ecommerce", "question": "Show all orders created in the last 30 days."},
    {"domain": "ecommerce", "question": "Find all products whose name starts with 'Premium'."},
    {
        "domain": "ecommerce",
        "question": "List all pending orders sorted by total amount descending.",
    },
    {"domain": "ecommerce", "question": "How many products are there in each category?"},
    {"domain": "ecommerce", "question": "Show the 5 cheapest products."},
    {"domain": "ecommerce", "question": "Find all customers created in 2023."},
    {"domain": "ecommerce", "question": "List all products with stock quantity greater than 100."},
    # HR (20)
    {"domain": "hr", "question": "List all employees sorted by salary descending."},
    {"domain": "hr", "question": "Show all departments and their locations."},
    {"domain": "hr", "question": "Find all employees with a salary above 80000."},
    {"domain": "hr", "question": "What are the names of departments located in 'New York'?"},
    {"domain": "hr", "question": "List all employees hired after 2022-01-01."},
    {"domain": "hr", "question": "Show all departments with a budget over 1000000."},
    {"domain": "hr", "question": "Find all employees who have no manager."},
    {"domain": "hr", "question": "List the 5 highest-paid employees."},
    {"domain": "hr", "question": "Show all performance reviews with a score of 5."},
    {"domain": "hr", "question": "Find all employees whose name contains 'Smith'."},
    {"domain": "hr", "question": "List all distinct department locations."},
    {"domain": "hr", "question": "Show all employees with a salary between 50000 and 80000."},
    {"domain": "hr", "question": "Find the employee with the longest tenure (earliest hire date)."},
    {"domain": "hr", "question": "List all departments sorted by budget ascending."},
    {"domain": "hr", "question": "Show all performance reviews from the year 2024."},
    {"domain": "hr", "question": "Find all employees hired in the last 6 months."},
    {"domain": "hr", "question": "What is the total budget across all departments?"},
    {"domain": "hr", "question": "List all employees with salary exactly equal to 75000."},
    {"domain": "hr", "question": "Show the 10 most recent performance reviews."},
    {"domain": "hr", "question": "Find all departments with 'Engineering' in their name."},
    # SaaS (20)
    {"domain": "saas", "question": "List all users on the 'enterprise' plan."},
    {"domain": "saas", "question": "Show all active subscriptions."},
    {"domain": "saas", "question": "Find all users who have not been active in the last 30 days."},
    {"domain": "saas", "question": "What are the plan names and MRR of all subscriptions?"},
    {"domain": "saas", "question": "List all events of type 'page_view'."},
    {"domain": "saas", "question": "Show all users created in 2024 sorted by creation date."},
    {"domain": "saas", "question": "Find all cancelled subscriptions."},
    {"domain": "saas", "question": "List the 10 most recent events."},
    {"domain": "saas", "question": "Show all subscriptions with MRR above 500."},
    {"domain": "saas", "question": "Find all users whose email ends with '@company.com'."},
    {"domain": "saas", "question": "List all distinct event types."},
    {"domain": "saas", "question": "Show all paused subscriptions sorted by started_at."},
    {"domain": "saas", "question": "Find the subscription with the highest MRR."},
    {
        "domain": "saas",
        "question": "List all users on the 'free' plan who signed up after 2023-06-01.",
    },
    {"domain": "saas", "question": "Show all events that occurred today."},
    {"domain": "saas", "question": "Find all subscriptions started in the last 90 days."},
    {"domain": "saas", "question": "List all users sorted by last active timestamp descending."},
    {"domain": "saas", "question": "Show all events for user with id 42."},
    {"domain": "saas", "question": "Find all subscriptions where cancelled_at is not null."},
    {"domain": "saas", "question": "What are the distinct plans that users are currently on?"},
]

# JOIN (70): 2-3 table JOINs with explicit JOIN syntax
# Inspired by Spider "medium" difficulty patterns
JOIN_PAIRS = [
    # E-commerce (24)
    {"domain": "ecommerce", "question": "Show all orders with customer names."},
    {"domain": "ecommerce", "question": "List all order items with product names and order IDs."},
    {"domain": "ecommerce", "question": "Find all customers who have placed at least one order."},
    {
        "domain": "ecommerce",
        "question": "Show the product name and quantity for each item in order 101.",
    },
    {"domain": "ecommerce", "question": "List all customers who have never placed an order."},
    {"domain": "ecommerce", "question": "Show each order with its customer name and total amount."},
    {"domain": "ecommerce", "question": "Find all products that have been ordered at least once."},
    {"domain": "ecommerce", "question": "List all products that have never been ordered."},
    {
        "domain": "ecommerce",
        "question": "Show the customer name, order id, and order status for all delivered orders.",
    },
    {
        "domain": "ecommerce",
        "question": "Find the names of customers who ordered products in the 'Books' category.",
    },
    {
        "domain": "ecommerce",
        "question": "List all order items with the product name, category, and unit price.",
    },
    {"domain": "ecommerce", "question": "Show customers and their most recent order date."},
    {
        "domain": "ecommerce",
        "question": "Find all orders that contain a product named 'Widget Pro'.",
    },
    {
        "domain": "ecommerce",
        "question": "List the customer name and total number of orders for each customer.",
    },
    {"domain": "ecommerce", "question": "Show products ordered by customer 'John Doe'."},
    {
        "domain": "ecommerce",
        "question": "Find the total revenue per product (sum of quantity times unit_price from order_items).",
    },
    {
        "domain": "ecommerce",
        "question": "List all customers who have orders with status 'pending' along with the order details.",
    },
    {"domain": "ecommerce", "question": "Show all orders with their item count."},
    {
        "domain": "ecommerce",
        "question": "Find customers who have ordered products from more than one category.",
    },
    {
        "domain": "ecommerce",
        "question": "List the product names and quantities for all items in cancelled orders.",
    },
    {
        "domain": "ecommerce",
        "question": "Show the customer name and the product names they have ordered.",
    },
    {
        "domain": "ecommerce",
        "question": "Find orders where the total amount exceeds the sum of item prices.",
    },
    {
        "domain": "ecommerce",
        "question": "List all products and the number of orders that include them.",
    },
    {
        "domain": "ecommerce",
        "question": "Show each customer's name and their total spend across all orders.",
    },
    # HR (23)
    {"domain": "hr", "question": "Show all employees with their department names."},
    {"domain": "hr", "question": "List employees and the names of their managers."},
    {"domain": "hr", "question": "Find all employees in the 'Engineering' department."},
    {
        "domain": "hr",
        "question": "Show each employee's name, department, and most recent review score.",
    },
    {"domain": "hr", "question": "List all departments with the number of employees in each."},
    {"domain": "hr", "question": "Find employees who have never received a performance review."},
    {
        "domain": "hr",
        "question": "Show the department name and average salary for each department.",
    },
    {"domain": "hr", "question": "List all employees whose manager earns more than 100000."},
    {
        "domain": "hr",
        "question": "Find all performance reviews along with the employee name and department.",
    },
    {"domain": "hr", "question": "Show departments that have no employees."},
    {
        "domain": "hr",
        "question": "List all employees who work in departments located in 'San Francisco'.",
    },
    {"domain": "hr", "question": "Find the reviewer name for each performance review."},
    {"domain": "hr", "question": "Show all employees and their department budgets."},
    {"domain": "hr", "question": "List employees who earn more than their manager."},
    {"domain": "hr", "question": "Find the total salary expense per department."},
    {
        "domain": "hr",
        "question": "Show each department and the name of the highest-paid employee in it.",
    },
    {"domain": "hr", "question": "List all employees along with their review count."},
    {
        "domain": "hr",
        "question": "Find departments where the total salary exceeds the department budget.",
    },
    {"domain": "hr", "question": "Show all employees hired after their manager was hired."},
    {"domain": "hr", "question": "List employees and the average review score they received."},
    {"domain": "hr", "question": "Find all departments with at least 5 employees."},
    {"domain": "hr", "question": "Show each employee with their department location and budget."},
    {
        "domain": "hr",
        "question": "List all performance reviews where the reviewer is from a different department than the employee.",
    },
    # SaaS (23)
    {"domain": "saas", "question": "Show all subscriptions with user email addresses."},
    {"domain": "saas", "question": "List all events with the user's plan information."},
    {"domain": "saas", "question": "Find all users who have an active subscription."},
    {"domain": "saas", "question": "Show each user's email and their subscription plan name."},
    {"domain": "saas", "question": "List users who have no subscriptions."},
    {"domain": "saas", "question": "Find all events for users on the 'enterprise' plan."},
    {
        "domain": "saas",
        "question": "Show each user's email, subscription MRR, and subscription status.",
    },
    {"domain": "saas", "question": "List users who have both active and cancelled subscriptions."},
    {"domain": "saas", "question": "Find the total MRR per plan name."},
    {"domain": "saas", "question": "Show all users and the count of events they generated."},
    {"domain": "saas", "question": "List users who have generated more than 100 events."},
    {
        "domain": "saas",
        "question": "Find all subscriptions for users who signed up in the last 30 days.",
    },
    {"domain": "saas", "question": "Show each user's most recent event type and timestamp."},
    {
        "domain": "saas",
        "question": "List the email and plan of users whose subscription MRR exceeds 1000.",
    },
    {"domain": "saas", "question": "Find users who have events but no subscription."},
    {
        "domain": "saas",
        "question": "Show all cancelled subscriptions with the user's email and plan.",
    },
    {
        "domain": "saas",
        "question": "List users whose last active timestamp is before their subscription start date.",
    },
    {
        "domain": "saas",
        "question": "Find the number of events per event type for users on the 'pro' plan.",
    },
    {
        "domain": "saas",
        "question": "Show each subscription with the user's signup date and last active date.",
    },
    {
        "domain": "saas",
        "question": "List all users who have events of type 'purchase' along with event details.",
    },
    {"domain": "saas", "question": "Find the average MRR for each user plan."},
    {
        "domain": "saas",
        "question": "Show users and the total number of subscriptions they have had.",
    },
    {
        "domain": "saas",
        "question": "List all users who have an active subscription but have not been active in the last 7 days.",
    },
]

# AGGREGATION (60): GROUP BY, HAVING, COUNT/SUM/AVG
# Inspired by Spider aggregation patterns
AGGREGATION_PAIRS = [
    # E-commerce (20)
    {"domain": "ecommerce", "question": "What is the total revenue from all orders?"},
    {"domain": "ecommerce", "question": "How many orders have been placed per status?"},
    {"domain": "ecommerce", "question": "What is the average order total amount?"},
    {"domain": "ecommerce", "question": "Find the total number of products per category."},
    {"domain": "ecommerce", "question": "What is the maximum and minimum product price?"},
    {"domain": "ecommerce", "question": "Show the number of orders placed per month in 2024."},
    {"domain": "ecommerce", "question": "Find categories with more than 10 products."},
    {"domain": "ecommerce", "question": "What is the total quantity of items sold per product?"},
    {"domain": "ecommerce", "question": "Show the average product price per category."},
    {"domain": "ecommerce", "question": "Find customers who have placed more than 5 orders."},
    {"domain": "ecommerce", "question": "What is the total revenue per customer?"},
    {
        "domain": "ecommerce",
        "question": "Show the number of distinct customers who placed orders each month.",
    },
    {
        "domain": "ecommerce",
        "question": "Find products with total sales quantity exceeding 100 units.",
    },
    {"domain": "ecommerce", "question": "What is the average number of items per order?"},
    {"domain": "ecommerce", "question": "Show the total stock quantity across all products."},
    {
        "domain": "ecommerce",
        "question": "Find the category with the highest average product price.",
    },
    {"domain": "ecommerce", "question": "What is the count of orders per day for the last 7 days?"},
    {
        "domain": "ecommerce",
        "question": "Show categories where the total stock value (price times quantity) exceeds 10000.",
    },
    {"domain": "ecommerce", "question": "Find the number of new customers per week in 2024."},
    {"domain": "ecommerce", "question": "What is the total revenue from delivered orders only?"},
    # HR (20)
    {"domain": "hr", "question": "What is the average salary across all employees?"},
    {"domain": "hr", "question": "How many employees are in each department?"},
    {"domain": "hr", "question": "What is the total salary expense per department?"},
    {"domain": "hr", "question": "Find departments with an average salary above 70000."},
    {"domain": "hr", "question": "Show the count of performance reviews per employee."},
    {"domain": "hr", "question": "What is the average review score per department?"},
    {"domain": "hr", "question": "Find employees with an average review score below 3."},
    {"domain": "hr", "question": "Show the number of employees hired per year."},
    {"domain": "hr", "question": "What is the maximum salary in each department?"},
    {
        "domain": "hr",
        "question": "Find departments where total salary exceeds the department budget.",
    },
    {"domain": "hr", "question": "Show the count of reviews given by each reviewer."},
    {"domain": "hr", "question": "What is the minimum and maximum hire date per department?"},
    {"domain": "hr", "question": "Find managers who manage more than 3 employees."},
    {"domain": "hr", "question": "Show the average salary per department location."},
    {
        "domain": "hr",
        "question": "What is the total number of performance reviews per month in 2024?",
    },
    {
        "domain": "hr",
        "question": "Find departments with more than 10 employees and average salary above 60000.",
    },
    {"domain": "hr", "question": "Show the salary range (max minus min) per department."},
    {
        "domain": "hr",
        "question": "What is the count of employees with no manager, grouped by department?",
    },
    {"domain": "hr", "question": "Find the number of reviews with score 5 per department."},
    {
        "domain": "hr",
        "question": "Show the total budget per location for locations with at least 2 departments.",
    },
    # SaaS (20)
    {"domain": "saas", "question": "What is the total MRR from all active subscriptions?"},
    {"domain": "saas", "question": "How many users are on each plan?"},
    {"domain": "saas", "question": "What is the average MRR per plan name?"},
    {"domain": "saas", "question": "Find plans with more than 100 active subscriptions."},
    {"domain": "saas", "question": "Show the number of events per event type."},
    {"domain": "saas", "question": "What is the total number of events per user?"},
    {"domain": "saas", "question": "Find users with more than 50 events in the last 30 days."},
    {"domain": "saas", "question": "Show the count of new subscriptions per month."},
    {
        "domain": "saas",
        "question": "What is the average time between user creation and first subscription?",
    },
    {"domain": "saas", "question": "Find event types with more than 1000 occurrences."},
    {
        "domain": "saas",
        "question": "Show the total MRR per plan name for active subscriptions only.",
    },
    {"domain": "saas", "question": "What is the count of cancelled subscriptions per month?"},
    {"domain": "saas", "question": "Find plans with average MRR above 200."},
    {
        "domain": "saas",
        "question": "Show the number of users who signed up each day in the last 30 days.",
    },
    {
        "domain": "saas",
        "question": "What is the total number of active vs paused vs cancelled subscriptions?",
    },
    {"domain": "saas", "question": "Find users who have subscriptions across more than one plan."},
    {
        "domain": "saas",
        "question": "Show the average MRR of cancelled subscriptions vs active subscriptions.",
    },
    {"domain": "saas", "question": "What is the count of distinct users per event type?"},
    {
        "domain": "saas",
        "question": "Find the month with the highest total MRR from new subscriptions.",
    },
    {"domain": "saas", "question": "Show the number of events per hour of the day."},
]

# CTE/SUBQUERY (50): WITH clause or nested subqueries
# Adapted from Spider hard/extra-hard patterns
CTE_SUBQUERY_PAIRS = [
    # E-commerce (17)
    {
        "domain": "ecommerce",
        "question": "Find customers whose total spend is above the average total spend across all customers.",
    },
    {
        "domain": "ecommerce",
        "question": "List products that have been ordered more times than the average product order count.",
    },
    {
        "domain": "ecommerce",
        "question": "Show the top 3 customers by total spend along with their order details.",
    },
    {
        "domain": "ecommerce",
        "question": "Find products that appear in orders from at least 5 different customers.",
    },
    {
        "domain": "ecommerce",
        "question": "With a CTE for monthly revenue, show months where revenue exceeded the overall monthly average.",
    },
    {
        "domain": "ecommerce",
        "question": "List customers who have ordered every product in the 'Electronics' category.",
    },
    {"domain": "ecommerce", "question": "Find the second most expensive product in each category."},
    {
        "domain": "ecommerce",
        "question": "Show customers whose average order amount is higher than the global average order amount.",
    },
    {
        "domain": "ecommerce",
        "question": "Using a CTE, calculate the month-over-month revenue growth rate.",
    },
    {
        "domain": "ecommerce",
        "question": "Find products that have never been ordered but are in a category that has at least one ordered product.",
    },
    {
        "domain": "ecommerce",
        "question": "List the top 5 products by total revenue, including the percentage of total revenue each represents.",
    },
    {
        "domain": "ecommerce",
        "question": "Find all customers who placed an order in every month of 2024.",
    },
    {
        "domain": "ecommerce",
        "question": "Show orders whose total amount is more than twice the customer's average order amount.",
    },
    {
        "domain": "ecommerce",
        "question": "Using a CTE, find categories where the most expensive product costs more than 3 times the category average.",
    },
    {
        "domain": "ecommerce",
        "question": "List customers who only buy products from a single category.",
    },
    {
        "domain": "ecommerce",
        "question": "Find the most popular product per category based on total quantity ordered.",
    },
    {
        "domain": "ecommerce",
        "question": "Show the running total of distinct customers who placed their first order each month.",
    },
    # HR (17)
    {
        "domain": "hr",
        "question": "Find employees whose salary is above the average salary in their department.",
    },
    {
        "domain": "hr",
        "question": "List departments where every employee has a review score of at least 3.",
    },
    {
        "domain": "hr",
        "question": "Show employees who earn more than their department's average but less than the company average.",
    },
    {
        "domain": "hr",
        "question": "Using a CTE, find departments where the total salary exceeds the department budget and the average review score is below 3.",
    },
    {
        "domain": "hr",
        "question": "Find the employee with the highest salary in each department, along with the department budget.",
    },
    {
        "domain": "hr",
        "question": "List managers whose direct reports have an average review score higher than the company average.",
    },
    {"domain": "hr", "question": "Show employees who were hired before their manager."},
    {
        "domain": "hr",
        "question": "Using a CTE, find departments where the median salary is above the company-wide median.",
    },
    {
        "domain": "hr",
        "question": "Find employees who have received reviews from more than 3 different reviewers.",
    },
    {
        "domain": "hr",
        "question": "List departments where the highest-paid employee earns more than twice the department average.",
    },
    {
        "domain": "hr",
        "question": "Show the hire date and salary of the most recently hired employee in each department.",
    },
    {
        "domain": "hr",
        "question": "Find employees whose review scores have improved over consecutive reviews.",
    },
    {
        "domain": "hr",
        "question": "Using a CTE for average department salary, list employees in departments with below-average salary budgets who earn above-average salaries.",
    },
    {
        "domain": "hr",
        "question": "Find all employees who are managers of managers (second-level managers).",
    },
    {
        "domain": "hr",
        "question": "Show departments where more than half of employees have a review score of 4 or above.",
    },
    {
        "domain": "hr",
        "question": "List the top 3 departments by average review score, along with their budget utilization ratio.",
    },
    {
        "domain": "hr",
        "question": "Find employees who earn more than every employee in a different department.",
    },
    # SaaS (16)
    {
        "domain": "saas",
        "question": "Find users whose total MRR across all subscriptions is above the platform average.",
    },
    {
        "domain": "saas",
        "question": "List plans where the average MRR of cancelled subscriptions exceeds the average MRR of active ones.",
    },
    {
        "domain": "saas",
        "question": "Show users who have more events than the average user event count.",
    },
    {
        "domain": "saas",
        "question": "Using a CTE, find the month with the highest number of new subscriptions and compare it to the average.",
    },
    {"domain": "saas", "question": "Find users who have subscriptions on every available plan."},
    {
        "domain": "saas",
        "question": "List users whose most recent event was more than 30 days ago but have an active subscription.",
    },
    {
        "domain": "saas",
        "question": "Show the plan name and count of users who upgraded from a lower-MRR plan to a higher-MRR plan.",
    },
    {
        "domain": "saas",
        "question": "Using a CTE, calculate the churn rate per month (cancelled subscriptions divided by total active at month start).",
    },
    {
        "domain": "saas",
        "question": "Find users who generated events of every event type present in the system.",
    },
    {
        "domain": "saas",
        "question": "List event types that are only used by users on the 'enterprise' plan.",
    },
    {
        "domain": "saas",
        "question": "Show users with the highest MRR in each plan, including the plan average MRR.",
    },
    {
        "domain": "saas",
        "question": "Using a CTE, find users whose activity (event count) in the last 30 days is less than half their activity in the 30 days before that.",
    },
    {
        "domain": "saas",
        "question": "Find subscriptions that were cancelled within 7 days of being started.",
    },
    {
        "domain": "saas",
        "question": "List plans where total MRR from active subscriptions exceeds the total from paused and cancelled combined.",
    },
    {
        "domain": "saas",
        "question": "Show the average time to cancellation per plan name for cancelled subscriptions.",
    },
    {
        "domain": "saas",
        "question": "Find users who have multiple active subscriptions on different plans.",
    },
]

# WINDOW FUNCTION (30): RANK, ROW_NUMBER, DENSE_RANK, LAG, LEAD
# Custom authored - Spider has none
WINDOW_FUNCTION_PAIRS = [
    # E-commerce (10)
    {
        "domain": "ecommerce",
        "question": "For each customer, show their order count and rank them by total spend descending.",
    },
    {
        "domain": "ecommerce",
        "question": "Show each order with the previous order's total amount for the same customer.",
    },
    {
        "domain": "ecommerce",
        "question": "Calculate the running total of order revenue by order date.",
    },
    {
        "domain": "ecommerce",
        "question": "For each product category, rank products by price and show the rank within category.",
    },
    {
        "domain": "ecommerce",
        "question": "Show each order with the next order's date for the same customer.",
    },
    {
        "domain": "ecommerce",
        "question": "Assign a row number to each order per customer sorted by order date.",
    },
    {
        "domain": "ecommerce",
        "question": "For each product, show its price and the average price in its category.",
    },
    {
        "domain": "ecommerce",
        "question": "Show the top 3 most expensive products per category using dense rank.",
    },
    {
        "domain": "ecommerce",
        "question": "For each order, show the percentage of total revenue it represents.",
    },
    {
        "domain": "ecommerce",
        "question": "Show a 3-order moving average of total_amount per customer.",
    },
    # HR (10)
    {"domain": "hr", "question": "Rank employees by salary within each department, highest first."},
    {
        "domain": "hr",
        "question": "Show each employee's hire date and the hire date of the employee hired just before them in the same department.",
    },
    {
        "domain": "hr",
        "question": "For each employee, show how their salary compares to the department average.",
    },
    {
        "domain": "hr",
        "question": "Assign a row number to employees within each department ordered by hire date.",
    },
    {
        "domain": "hr",
        "question": "Show each performance review score alongside the previous review score for the same employee.",
    },
    {
        "domain": "hr",
        "question": "For each department, show the cumulative salary budget as employees are added by hire date.",
    },
    {"domain": "hr", "question": "Show the top 2 highest-paid employees per department."},
    {
        "domain": "hr",
        "question": "Show each employee's salary as a percentile within their department.",
    },
    {
        "domain": "hr",
        "question": "For each employee, show the next scheduled review date after their most recent review.",
    },
    {
        "domain": "hr",
        "question": "Show each employee's salary and the maximum salary in their department.",
    },
    # SaaS (10)
    {
        "domain": "saas",
        "question": "For each user, show their subscription MRR and rank among users on the same plan.",
    },
    {
        "domain": "saas",
        "question": "Show monthly revenue with the previous month's revenue using LAG.",
    },
    {
        "domain": "saas",
        "question": "Calculate the running total of MRR sorted by subscription start date.",
    },
    {
        "domain": "saas",
        "question": "For each user, show their event count and rank users by total events descending.",
    },
    {
        "domain": "saas",
        "question": "Show each event with the time of the next event for the same user.",
    },
    {
        "domain": "saas",
        "question": "Assign a sequence number to each subscription per user ordered by start date.",
    },
    {
        "domain": "saas",
        "question": "For each active subscription, show the MRR and the average MRR across all active subscriptions on the same plan.",
    },
    {
        "domain": "saas",
        "question": "Show the top 3 most active users by event count using dense rank.",
    },
    {
        "domain": "saas",
        "question": "For each user, show the percentage of total MRR their subscription represents.",
    },
    {"domain": "saas", "question": "Show a 7-day moving average of daily event counts."},
]

# UNANSWERABLE (30): Questions referencing non-existent columns/tables
# Custom authored
UNANSWERABLE_PAIRS = [
    # E-commerce (10)
    {
        "domain": "ecommerce",
        "question": "What is the average customer lifetime value?",
        "expected_error": "Cannot answer: The schema does not contain a customer lifetime value (LTV) column, and computing it would require revenue attribution logic not derivable from orders alone.",
    },
    {
        "domain": "ecommerce",
        "question": "Show customer satisfaction ratings for each product.",
        "expected_error": "Cannot answer: The schema has no ratings or reviews table. Product feedback data is not available.",
    },
    {
        "domain": "ecommerce",
        "question": "List all customers who used a discount code on their last order.",
        "expected_error": "Cannot answer: There is no discount_code or coupon column in the orders or order_items tables.",
    },
    {
        "domain": "ecommerce",
        "question": "What is the return rate for each product category?",
        "expected_error": "Cannot answer: The schema has no returns table or return_status column. Order returns are not tracked.",
    },
    {
        "domain": "ecommerce",
        "question": "Show the shipping address for each order.",
        "expected_error": "Cannot answer: The orders table does not contain a shipping_address column, and there is no addresses table.",
    },
    {
        "domain": "ecommerce",
        "question": "Which marketing campaign drove the most orders?",
        "expected_error": "Cannot answer: There is no campaigns table or campaign_id/utm_source column linking orders to marketing campaigns.",
    },
    {
        "domain": "ecommerce",
        "question": "List the top suppliers by number of products supplied.",
        "expected_error": "Cannot answer: The schema has no suppliers table or supplier_id column in the products table.",
    },
    {
        "domain": "ecommerce",
        "question": "What is the average delivery time by shipping carrier?",
        "expected_error": "Cannot answer: There is no shipping_carrier or delivery_date column. The schema tracks order status but not carrier-level delivery metrics.",
    },
    {
        "domain": "ecommerce",
        "question": "Show product page views for each product this month.",
        "expected_error": "Cannot answer: The schema does not contain a page_views or analytics table. Web traffic data is not available.",
    },
    {
        "domain": "ecommerce",
        "question": "Calculate the cart abandonment rate.",
        "expected_error": "Cannot answer: There is no shopping_carts or cart_items table. The schema only tracks completed orders, not abandoned carts.",
    },
    # HR (10)
    {
        "domain": "hr",
        "question": "List employees with their LinkedIn profile URLs.",
        "expected_error": "Cannot answer: The employees table does not have a linkedin_url or social_profiles column.",
    },
    {
        "domain": "hr",
        "question": "Show the educational background of each employee.",
        "expected_error": "Cannot answer: There is no education table or degree/university column in the employees table.",
    },
    {
        "domain": "hr",
        "question": "What are the benefit plan costs per department?",
        "expected_error": "Cannot answer: The schema has no benefits or benefit_plans table. Employee benefit data is not tracked.",
    },
    {
        "domain": "hr",
        "question": "Show employee attendance records for last quarter.",
        "expected_error": "Cannot answer: There is no attendance or time_tracking table in the schema.",
    },
    {
        "domain": "hr",
        "question": "List all open job requisitions by department.",
        "expected_error": "Cannot answer: The schema has no job_requisitions or open_positions table. Hiring pipeline data is not available.",
    },
    {
        "domain": "hr",
        "question": "What is the employee turnover rate by department?",
        "expected_error": "Cannot answer: There is no termination_date or employment_status column in the employees table to calculate turnover.",
    },
    {
        "domain": "hr",
        "question": "Show each employee's training certifications.",
        "expected_error": "Cannot answer: The schema does not include a certifications or training_records table.",
    },
    {
        "domain": "hr",
        "question": "Calculate overtime hours per employee this month.",
        "expected_error": "Cannot answer: There is no timesheets or work_hours table. The schema does not track hours worked.",
    },
    {
        "domain": "hr",
        "question": "List employees eligible for retirement.",
        "expected_error": "Cannot answer: The employees table has no birth_date or age column, so retirement eligibility cannot be determined.",
    },
    {
        "domain": "hr",
        "question": "Show the promotion history for each employee.",
        "expected_error": "Cannot answer: There is no promotions or job_history table. The schema does not track role changes over time.",
    },
    # SaaS (10)
    {
        "domain": "saas",
        "question": "Show churn rate by acquisition channel.",
        "expected_error": "Cannot answer: There is no acquisition_channel or utm_source column in the users or subscriptions table.",
    },
    {
        "domain": "saas",
        "question": "What is the average Net Promoter Score by plan?",
        "expected_error": "Cannot answer: The schema has no nps_scores or survey_responses table. NPS data is not tracked.",
    },
    {
        "domain": "saas",
        "question": "List users who submitted support tickets this week.",
        "expected_error": "Cannot answer: There is no support_tickets or help_desk table in the schema.",
    },
    {
        "domain": "saas",
        "question": "Show feature usage breakdown by user plan.",
        "expected_error": "Cannot answer: While the events table tracks event_type, there is no features table or feature_id to map events to specific product features.",
    },
    {
        "domain": "saas",
        "question": "Calculate the average time to first value for new users.",
        "expected_error": "Cannot answer: There is no onboarding_milestones or first_value_event definition in the schema to measure time to first value.",
    },
    {
        "domain": "saas",
        "question": "Show revenue by geographic region.",
        "expected_error": "Cannot answer: The schema has no country, region, or geographic location columns in any table.",
    },
    {
        "domain": "saas",
        "question": "List all API integrations configured per user.",
        "expected_error": "Cannot answer: There is no integrations or api_keys table. Third-party integration data is not tracked.",
    },
    {
        "domain": "saas",
        "question": "What is the average response time of our API endpoints?",
        "expected_error": "Cannot answer: The schema does not contain an api_logs or performance_metrics table. API performance data is not available.",
    },
    {
        "domain": "saas",
        "question": "Show user referral chains and their conversion rates.",
        "expected_error": "Cannot answer: There is no referrals table or referred_by column in the users table.",
    },
    {
        "domain": "saas",
        "question": "Calculate the cost per acquisition by marketing channel.",
        "expected_error": "Cannot answer: The schema has no marketing_spend, campaigns, or acquisition_channel columns. Marketing cost data is not tracked.",
    },
]


# ---------------------------------------------------------------------------
# Build full dataset of 300 pairs (prompts only, no model responses yet)
# ---------------------------------------------------------------------------
def build_all_pairs() -> list[dict]:
    """Build all 300 NL/SQL pairs."""
    pairs = []

    # Simple (60)
    for p in SIMPLE_PAIRS:
        schema_ddl = SCHEMAS[p["domain"]]
        prompt = build_prompt(schema_ddl, p["question"])
        pairs.append(
            {
                "prompt": prompt,
                "question": p["question"],
                "domain": p["domain"],
                "variant": "simple",
                "source": "custom_authored_spider_inspired",
            }
        )

    # Join (70)
    for p in JOIN_PAIRS:
        schema_ddl = SCHEMAS[p["domain"]]
        prompt = build_prompt(schema_ddl, p["question"])
        pairs.append(
            {
                "prompt": prompt,
                "question": p["question"],
                "domain": p["domain"],
                "variant": "join",
                "source": "custom_authored_spider_inspired",
            }
        )

    # Aggregation (60)
    for p in AGGREGATION_PAIRS:
        schema_ddl = SCHEMAS[p["domain"]]
        prompt = build_prompt(schema_ddl, p["question"])
        pairs.append(
            {
                "prompt": prompt,
                "question": p["question"],
                "domain": p["domain"],
                "variant": "aggregation",
                "source": "custom_authored_spider_inspired",
            }
        )

    # CTE/Subquery (50)
    for p in CTE_SUBQUERY_PAIRS:
        schema_ddl = SCHEMAS[p["domain"]]
        prompt = build_prompt(schema_ddl, p["question"])
        pairs.append(
            {
                "prompt": prompt,
                "question": p["question"],
                "domain": p["domain"],
                "variant": "cte_subquery",
                "source": "custom_authored_spider_inspired",
            }
        )

    # Window functions (30)
    for p in WINDOW_FUNCTION_PAIRS:
        schema_ddl = SCHEMAS[p["domain"]]
        prompt = build_prompt(schema_ddl, p["question"])
        pairs.append(
            {
                "prompt": prompt,
                "question": p["question"],
                "domain": p["domain"],
                "variant": "window_function",
                "source": "custom_authored",
            }
        )

    # Unanswerable (30)
    for p in UNANSWERABLE_PAIRS:
        schema_ddl = SCHEMAS[p["domain"]]
        prompt = build_prompt(schema_ddl, p["question"])
        pairs.append(
            {
                "prompt": prompt,
                "question": p["question"],
                "expected_error": p["expected_error"],
                "domain": p["domain"],
                "variant": "unanswerable",
                "source": "custom_authored",
            }
        )

    print(f"Total pairs built: {len(pairs)}")

    # Verify distribution
    variant_counts = {}
    for p in pairs:
        v = p["variant"]
        variant_counts[v] = variant_counts.get(v, 0) + 1
    print(f"Variant distribution: {variant_counts}")

    domain_counts = {}
    for p in pairs:
        d = p["domain"]
        domain_counts[d] = domain_counts.get(d, 0) + 1
    print(f"Domain distribution: {domain_counts}")

    return pairs


# ---------------------------------------------------------------------------
# Model calling with cost tracking
# ---------------------------------------------------------------------------
class CostTracker:
    def __init__(self):
        self.costs: dict[str, dict[str, float]] = {}

    def add(self, model: str, phase: str, cost: float):
        if model not in self.costs:
            self.costs[model] = {"tuning": 0.0, "production": 0.0}
        self.costs[model][phase] += cost

    def summary(self) -> list[dict]:
        result = []
        for model, phases in self.costs.items():
            result.append(
                {
                    "model": model,
                    "tuning_cost_usd": round(phases["tuning"], 6),
                    "production_cost_usd": round(phases["production"], 6),
                    "total_cost_usd": round(phases["tuning"] + phases["production"], 6),
                    "pairs_generated": 300,
                }
            )
        return result


def call_model(
    model: str,
    api_key: str,
    prompt: str,
    tracker: CostTracker,
    system_prompt: str = SYSTEM_PROMPT,
    phase: str = "production",
    max_retries: int = 5,
) -> tuple[str, float]:
    """Call an LLM model with retry logic. Returns (response_text, cost)."""
    for attempt in range(max_retries):
        try:
            resp = completion(
                model=model,
                api_key=api_key,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            text = resp.choices[0].message.content.strip()
            cost = resp._hidden_params.get("response_cost", 0.0) or 0.0
            tracker.add(model, phase, cost)
            return text, cost
        except Exception as e:
            err_str = str(e)
            if "rate_limit" in err_str.lower():
                wait = min(60, 5 * (attempt + 1))
                print(f"  Rate limited, waiting {wait}s... (attempt {attempt + 1})")
                time.sleep(wait)
            elif attempt < max_retries - 1:
                wait = 2**attempt
                print(f"  Retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(wait)
            else:
                print(f"  FAILED after {max_retries} attempts: {e}")
                return f'{{"error": "API call failed: {str(e)}"}}', 0.0


def validate_sql_syntax(sql_text: str) -> bool:
    """Check if SQL text is syntactically parseable."""
    if sql_text.strip().startswith("{"):
        try:
            data = json.loads(sql_text)
            return "error" in data
        except json.JSONDecodeError:
            return False

    parsed = sqlparse.parse(sql_text)
    if not parsed or len(parsed) == 0:
        return False
    stmt = parsed[0]
    tokens = [t for t in stmt.tokens if not t.is_whitespace]
    return len(tokens) > 0


def clean_response(text: str) -> str:
    """Clean model response: strip markdown fences, extra whitespace."""
    text = text.strip()
    if text.startswith("```sql"):
        text = text[6:]
    elif text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    # Remove json language tag prefix
    if text.startswith("json\n"):
        text = text[5:]
    elif text.startswith("json\r\n"):
        text = text[6:]
    # Remove trailing semicolons for consistency
    if text.endswith(";") and not text.strip().startswith("{"):
        text = text[:-1].strip()
    return text


# ---------------------------------------------------------------------------
# Prompt tuning (Step 4)
# ---------------------------------------------------------------------------
def run_tuning(pairs: list[dict], tracker: CostTracker) -> None:
    """Run prompt tuning on 30 mixed samples using GPT-4o."""
    import random

    print("\n=== Step 4: Prompt Tuning (30 samples with GPT-4o) ===")

    # Select 5 per variant
    by_variant: dict[str, list[int]] = {}
    for i, p in enumerate(pairs):
        v = p["variant"]
        if v not in by_variant:
            by_variant[v] = []
        by_variant[v].append(i)

    random.seed(99)
    tuning_samples = []
    for variant in DISTRIBUTION:
        indices = by_variant.get(variant, [])
        sample_size = min(5, len(indices))
        chosen = random.sample(indices, sample_size)
        tuning_samples.extend(chosen)

    print(f"Tuning on {len(tuning_samples)} samples...")

    api_key = os.environ["OPENROUTER_API_KEY"]
    model = "openrouter/openai/gpt-4o"

    valid_sql = 0
    valid_error = 0
    total_answerable = 0
    total_unanswerable = 0
    window_with_over = 0
    total_window = 0

    for idx in tuning_samples:
        pair = pairs[idx]
        prompt = pair["prompt"]
        variant = pair["variant"]

        sys_prompt = SYSTEM_PROMPT_WINDOW if variant == "window_function" else SYSTEM_PROMPT
        response, cost = call_model(
            model, api_key, prompt, tracker, system_prompt=sys_prompt, phase="tuning"
        )
        response = clean_response(response)

        if variant == "unanswerable":
            total_unanswerable += 1
            try:
                data = json.loads(response)
                if "error" in data:
                    valid_error += 1
                else:
                    print(f"  [TUNE FAIL] Unanswerable but no error key: {response[:100]}")
            except json.JSONDecodeError:
                print(f"  [TUNE FAIL] Unanswerable but not JSON: {response[:100]}")
        else:
            total_answerable += 1
            if validate_sql_syntax(response):
                valid_sql += 1
            else:
                print(f"  [TUNE FAIL] Invalid SQL: {response[:100]}")

            if variant == "window_function":
                total_window += 1
                if "OVER" in response.upper():
                    window_with_over += 1
                else:
                    print(f"  [TUNE WARN] Window function without OVER: {response[:80]}")

        time.sleep(0.2)

    sql_rate = valid_sql / total_answerable * 100 if total_answerable > 0 else 0
    error_rate = valid_error / total_unanswerable * 100 if total_unanswerable > 0 else 0

    print(f"\nTuning Results:")
    print(f"  SQL validity:     {valid_sql}/{total_answerable} ({sql_rate:.1f}%)")
    print(f"  Error JSON:       {valid_error}/{total_unanswerable} ({error_rate:.1f}%)")
    print(f"  Window w/ OVER:   {window_with_over}/{total_window}")
    print(f"  Tuning cost:      ${tracker.costs.get(model, {}).get('tuning', 0):.4f}")


# ---------------------------------------------------------------------------
# Full production run (Step 5)
# ---------------------------------------------------------------------------
def run_production(
    pairs: list[dict],
    model: str,
    api_key: str,
    output_path: Path,
    tracker: CostTracker,
    source_model_label: str,
) -> None:
    """Run full production for one model, writing JSONL incrementally."""
    print(f"\n=== Production Run: {source_model_label} -> {output_path.name} ===")

    # Check for existing records to support resume
    existing_count = 0
    if output_path.exists():
        with open(output_path) as f:
            existing_count = sum(1 for _ in f)
        if existing_count > 0:
            print(f"  Resuming from record {existing_count}")

    mode = "a" if existing_count > 0 else "w"
    valid_count = 0
    error_count = 0
    fail_count = 0
    window_over_count = 0

    with open(output_path, mode) as fout:
        for i, pair in enumerate(pairs):
            if i < existing_count:
                continue

            prompt = pair["prompt"]
            variant = pair["variant"]

            sys_prompt = SYSTEM_PROMPT_WINDOW if variant == "window_function" else SYSTEM_PROMPT
            response, cost = call_model(model, api_key, prompt, tracker, system_prompt=sys_prompt)
            response = clean_response(response)

            # Build metadata
            metadata: dict[str, Any] = {
                "dataset": "sql_generation",
                "variant": variant,
                "domain": pair.get("domain", "unknown"),
            }
            if pair.get("source"):
                metadata["source"] = pair["source"]

            record = {
                "prompt": prompt,
                "response": response,
                "source_model": source_model_label,
                "metadata": metadata,
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            # Track quality
            if variant == "unanswerable":
                try:
                    data = json.loads(response)
                    if "error" in data:
                        error_count += 1
                    else:
                        fail_count += 1
                except json.JSONDecodeError:
                    fail_count += 1
            else:
                if validate_sql_syntax(response):
                    valid_count += 1
                else:
                    fail_count += 1

                if variant == "window_function" and "OVER" in response.upper():
                    window_over_count += 1

            # Progress
            total_done = i + 1
            if total_done % 50 == 0 or total_done == len(pairs):
                answerable = sum(1 for p in pairs[:total_done] if p["variant"] != "unanswerable")
                unanswerable = total_done - answerable
                current_cost = tracker.costs.get(model, {}).get("production", 0)
                print(
                    f"  [{total_done}/{len(pairs)}] "
                    f"valid_sql={valid_count}/{answerable} "
                    f"valid_error={error_count}/{unanswerable} "
                    f"window_over={window_over_count} "
                    f"fails={fail_count} "
                    f"cost=${current_cost:.4f}"
                )

            # Rate limiting
            time.sleep(0.15)

    answerable_total = sum(1 for p in pairs if p["variant"] != "unanswerable")
    unanswerable_total = len(pairs) - answerable_total
    wf_total = sum(1 for p in pairs if p["variant"] == "window_function")
    print(
        f"\n  Final: valid_sql={valid_count}/{answerable_total}, "
        f"valid_error={error_count}/{unanswerable_total}, "
        f"window_over={window_over_count}/{wf_total}, "
        f"fails={fail_count}"
    )


# ---------------------------------------------------------------------------
# Post-processing: fix any remaining issues
# ---------------------------------------------------------------------------
def postprocess(output_path: Path, model: str, api_key: str, tracker: CostTracker) -> None:
    """Fix unanswerable responses that didn't return error JSON, and window functions without OVER."""
    print(f"\n=== Post-processing: {output_path.name} ===")

    records = []
    with open(output_path) as f:
        for line in f:
            records.append(json.loads(line))

    fixes = 0

    for i, rec in enumerate(records):
        variant = rec["metadata"]["variant"]
        response = rec["response"]

        # Fix json prefix
        if response.startswith("json\n") or response.startswith("json\r\n"):
            response = response.replace("json\n", "", 1).replace("json\r\n", "", 1).strip()
            records[i]["response"] = response
            fixes += 1

        # Fix unanswerable that didn't return error JSON
        if variant == "unanswerable":
            is_valid_error = False
            try:
                d = json.loads(response)
                is_valid_error = "error" in d
            except (json.JSONDecodeError, ValueError):
                pass

            if not is_valid_error:
                # Re-call with stronger prompt
                prompt = rec["prompt"]
                question = prompt.split("Write a PostgreSQL query to answer this question:\n")[
                    -1
                ].strip()
                domain = rec["metadata"].get("domain", "unknown")

                new_resp, _ = call_model(
                    model,
                    api_key,
                    prompt,
                    tracker,
                    system_prompt=SYSTEM_PROMPT.replace(
                        "If the question CANNOT", "CRITICAL: If the question CANNOT"
                    ),
                )
                new_resp = clean_response(new_resp)

                is_valid = False
                try:
                    d = json.loads(new_resp)
                    is_valid = "error" in d
                except (json.JSONDecodeError, ValueError):
                    pass

                if is_valid:
                    records[i]["response"] = new_resp
                else:
                    # Force error response
                    forced = json.dumps(
                        {
                            "error": f"Cannot answer: The question references data not present in the {domain} schema."
                        }
                    )
                    records[i]["response"] = forced
                fixes += 1
                time.sleep(1.5)

        # Fix window functions without OVER
        if variant == "window_function" and "OVER" not in response.upper():
            prompt = rec["prompt"]
            new_resp, _ = call_model(
                model, api_key, prompt, tracker, system_prompt=SYSTEM_PROMPT_WINDOW
            )
            new_resp = clean_response(new_resp)

            if "OVER" in new_resp.upper():
                records[i]["response"] = new_resp
            else:
                # Try with hint
                hint_prompt = prompt + "\n\nHint: Use PostgreSQL window functions (OVER clause)."
                new_resp2, _ = call_model(
                    model, api_key, hint_prompt, tracker, system_prompt=SYSTEM_PROMPT_WINDOW
                )
                new_resp2 = clean_response(new_resp2)
                if "OVER" in new_resp2.upper():
                    records[i]["response"] = new_resp2
                else:
                    records[i]["response"] = new_resp  # Use the re-call even without OVER
            fixes += 1
            time.sleep(0.5)

    # Write back
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  Total fixes: {fixes}")


# ---------------------------------------------------------------------------
# Quality spot-check
# ---------------------------------------------------------------------------
def spot_check(output_path: Path) -> dict[str, int]:
    """Spot-check and return quality metrics."""
    print(f"\n=== Spot-Check: {output_path.name} ===")

    records = []
    with open(output_path) as f:
        for line in f:
            records.append(json.loads(line))

    metrics = {
        "total": len(records),
        "valid_sql": 0,
        "valid_error": 0,
        "window_over": 0,
        "cte_pattern": 0,
        "issues": 0,
    }

    by_variant: dict[str, list[dict]] = {}
    for r in records:
        v = r["metadata"]["variant"]
        if v not in by_variant:
            by_variant[v] = []
        by_variant[v].append(r)

    for variant, vrecs in sorted(by_variant.items()):
        samples = vrecs[:5]
        print(f"\n  Variant: {variant} ({len(vrecs)} records)")
        for i, rec in enumerate(samples):
            response = rec["response"]
            if variant == "unanswerable":
                try:
                    data = json.loads(response)
                    if "error" in data and "Cannot answer" in data["error"]:
                        print(f"    [{i}] OK: {data['error'][:80]}")
                    else:
                        print(f"    [{i}] ISSUE: {response[:80]}")
                        metrics["issues"] += 1
                except json.JSONDecodeError:
                    print(f"    [{i}] ISSUE: not JSON: {response[:80]}")
                    metrics["issues"] += 1
            elif variant == "window_function":
                if "OVER" in response.upper():
                    print(f"    [{i}] OK: has OVER clause")
                else:
                    print(f"    [{i}] WARN: no OVER: {response[:80]}")
            elif variant == "cte_subquery":
                upper = response.upper()
                if "WITH " in upper or upper.count("SELECT") > 1:
                    print(f"    [{i}] OK: has CTE/subquery")
                else:
                    print(f"    [{i}] WARN: simple SQL: {response[:80]}")
            else:
                if validate_sql_syntax(response):
                    print(f"    [{i}] OK: valid SQL")
                else:
                    print(f"    [{i}] ISSUE: invalid: {response[:80]}")
                    metrics["issues"] += 1

    # Full counts
    for r in records:
        v = r["metadata"]["variant"]
        resp = r["response"]
        if v == "unanswerable":
            try:
                d = json.loads(resp)
                if "error" in d:
                    metrics["valid_error"] += 1
            except:
                pass
        else:
            if validate_sql_syntax(resp):
                metrics["valid_sql"] += 1
            if v == "window_function" and "OVER" in resp.upper():
                metrics["window_over"] += 1
            if v == "cte_subquery":
                upper = resp.upper()
                if "WITH " in upper or upper.count("SELECT") > 1:
                    metrics["cte_pattern"] += 1

    answerable = metrics["total"] - len(by_variant.get("unanswerable", []))
    unanswerable = len(by_variant.get("unanswerable", []))
    wf = len(by_variant.get("window_function", []))
    cte = len(by_variant.get("cte_subquery", []))

    print(f"\n  Summary:")
    print(
        f"    SQL validity:     {metrics['valid_sql']}/{answerable} ({100 * metrics['valid_sql'] / answerable:.1f}%)"
    )
    print(
        f"    Error validity:   {metrics['valid_error']}/{unanswerable} ({100 * metrics['valid_error'] / unanswerable:.1f}%)"
    )
    print(f"    Window w/ OVER:   {metrics['window_over']}/{wf}")
    print(f"    CTE/subquery:     {metrics['cte_pattern']}/{cte}")
    print(f"    Issues:           {metrics['issues']}")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    load_dotenv(ENV_PATH)

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if not openrouter_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)
    if not anthropic_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    print("API keys loaded successfully.")

    tracker = CostTracker()

    # Steps 1-3: Build all 300 pairs
    pairs = build_all_pairs()
    assert len(pairs) == 300, f"Expected 300 pairs, got {len(pairs)}"

    # Step 4: Prompt tuning
    run_tuning(pairs, tracker)

    # Step 5: Full production runs
    gpt4o_path = OUTPUT_DIR / "sql_generation_gpt4o.jsonl"
    haiku_path = OUTPUT_DIR / "sql_generation_haiku.jsonl"

    # GPT-4o
    run_production(
        pairs=pairs,
        model="openrouter/openai/gpt-4o",
        api_key=openrouter_key,
        output_path=gpt4o_path,
        tracker=tracker,
        source_model_label="openai/gpt-4o",
    )

    # Post-process GPT-4o
    postprocess(gpt4o_path, "openrouter/openai/gpt-4o", openrouter_key, tracker)

    # Haiku
    run_production(
        pairs=pairs,
        model="anthropic/claude-haiku-4-5-20251001",
        api_key=anthropic_key,
        output_path=haiku_path,
        tracker=tracker,
        source_model_label="anthropic/claude-haiku-4-5-20251001",
    )

    # Post-process Haiku
    postprocess(haiku_path, "anthropic/claude-haiku-4-5-20251001", anthropic_key, tracker)

    # Step 6: Write cost summary
    cost_summary = tracker.summary()
    cost_path = OUTPUT_DIR / "cost_summary.json"
    with open(cost_path, "w") as f:
        json.dump(cost_summary, f, indent=2)
    print(f"\nCost summary written to {cost_path}")
    print(json.dumps(cost_summary, indent=2))

    # Spot-checks
    spot_check(gpt4o_path)
    spot_check(haiku_path)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
