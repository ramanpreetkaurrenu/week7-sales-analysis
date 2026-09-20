import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# FOLDER SETUP
# --------------------------------------------------

DATA_FILE = "data/sales_data.csv"

OUTPUT_FOLDER = "outputs"
CHART_FOLDER = "outputs/charts"
REPORT_FOLDER = "outputs/reports"

os.makedirs(CHART_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

def load_data(file_path):
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)

        elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
            df = pd.read_excel(file_path)

        else:
            print("Unsupported file format.")
            return None

        print("\nData loaded successfully!")
        print(f"Rows: {df.shape[0]}")
        print(f"Columns: {df.shape[1]}")

        return df

    except FileNotFoundError:
        print("File not found.")
        return None

    except Exception as e:
        print("Error:", e)
        return None


# --------------------------------------------------
# 2. DATA EXPLORATION
# --------------------------------------------------

def explore_data(df):

    print("\n========== DATASET INFORMATION ==========")

    print("\nFirst 5 records:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate records:")
    print(df.duplicated().sum())

    print("\nStatistical summary:")
    print(df.describe(include="all"))


# --------------------------------------------------
# 3. DATA CLEANING
# --------------------------------------------------

def clean_data(df):

    print("\n========== CLEANING DATA ==========")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Convert date
    df["Order_Date"] = pd.to_datetime(
        df["Order_Date"],
        errors="coerce"
    )

    # Convert numerical columns
    numeric_columns = [
        "Quantity",
        "Unit_Price"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Fill missing numeric values
    for column in numeric_columns:
        df[column] = df[column].fillna(
            df[column].median()
        )

    # Fill categorical missing values
    categorical_columns = [
        "Customer_ID",
        "Product",
        "Category",
        "Region"
    ]

    for column in categorical_columns:
        df[column] = df[column].fillna("Unknown")

    # Remove rows where date is invalid
    df = df.dropna(subset=["Order_Date"])

    # Create Sales column
    df["Sales"] = df["Quantity"] * df["Unit_Price"]

    # Create Month column
    df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)

    print("Data cleaning completed.")
    print("Final rows:", len(df))

    return df


# --------------------------------------------------
# 4. BASIC METRICS
# --------------------------------------------------

def basic_metrics(df):

    total_sales = df["Sales"].sum()

    average_sale = df["Sales"].mean()

    total_orders = df["Order_ID"].nunique()

    total_quantity = df["Quantity"].sum()

    average_order_value = (
        total_sales / total_orders
    )

    print("\n========== BASIC METRICS ==========")

    print(f"Total Sales: ₹{total_sales:,.2f}")
    print(f"Average Sale: ₹{average_sale:,.2f}")
    print(f"Total Orders: {total_orders}")
    print(f"Total Quantity Sold: {total_quantity}")
    print(
        f"Average Order Value: ₹{average_order_value:,.2f}"
    )

    return {
        "Total Sales": total_sales,
        "Average Sale": average_sale,
        "Total Orders": total_orders,
        "Total Quantity": total_quantity,
        "Average Order Value": average_order_value
    }


# --------------------------------------------------
# 5. CATEGORY ANALYSIS
# --------------------------------------------------

def category_analysis(df):

    result = (
        df.groupby("Category")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\n========== SALES BY CATEGORY ==========")
    print(result)

    return result


# --------------------------------------------------
# 6. PRODUCT ANALYSIS
# --------------------------------------------------

def product_analysis(df):

    result = (
        df.groupby("Product")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\n========== TOP PRODUCTS ==========")
    print(result)

    print("\nTop 5 Products:")
    print(result.head(5))

    return result


# --------------------------------------------------
# 7. MONTHLY SALES
# --------------------------------------------------

def monthly_sales(df):

    result = (
        df.groupby("Month")["Sales"]
        .sum()
        .sort_index()
    )

    print("\n========== MONTHLY SALES ==========")
    print(result)

    return result


# --------------------------------------------------
# 8. MONTHLY GROWTH
# --------------------------------------------------

def monthly_growth(df):

    monthly = monthly_sales(df)

    growth = monthly.pct_change() * 100

    result = pd.DataFrame({
        "Sales": monthly,
        "Growth (%)": growth
    })

    print("\n========== MONTHLY GROWTH ==========")
    print(result)

    return result


# --------------------------------------------------
# 9. REGION ANALYSIS
# --------------------------------------------------

def region_analysis(df):

    result = (
        df.groupby("Region")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\n========== SALES BY REGION ==========")
    print(result)

    return result


# --------------------------------------------------
# 10. CUSTOMER ANALYSIS
# --------------------------------------------------

def customer_analysis(df):

    result = (
        df.groupby("Customer_ID")
        .agg(
            Total_Sales=("Sales", "sum"),
            Orders=("Order_ID", "nunique"),
            Quantity=("Quantity", "sum")
        )
        .sort_values(
            "Total_Sales",
            ascending=False
        )
    )

    print("\n========== CUSTOMER ANALYSIS ==========")
    print(result.head(10))

    return result


# --------------------------------------------------
# 11. CUSTOMER LIFETIME VALUE
# --------------------------------------------------

def customer_lifetime_value(df):

    customer = (
        df.groupby("Customer_ID")
        .agg(
            Total_Sales=("Sales", "sum"),
            Orders=("Order_ID", "nunique")
        )
    )

    customer["Average_Order_Value"] = (
        customer["Total_Sales"] /
        customer["Orders"]
    )

    print("\n========== CUSTOMER LIFETIME VALUE ==========")
    print(customer.sort_values(
        "Total_Sales",
        ascending=False
    ).head(10))

    return customer


# --------------------------------------------------
# 12. PEAK SALES PERIOD
# --------------------------------------------------

def peak_sales(df):

    daily = (
        df.groupby("Order_Date")["Sales"]
        .sum()
    )

    peak_date = daily.idxmax()
    peak_value = daily.max()

    print("\n========== PEAK SALES ==========")

    print("Peak Sales Date:", peak_date)
    print(f"Peak Sales: ₹{peak_value:,.2f}")

    return peak_date, peak_value


# --------------------------------------------------
# 13. MOVING AVERAGE FORECAST
# --------------------------------------------------

def sales_forecast(df):

    monthly = monthly_sales(df)

    forecast = monthly.rolling(
        window=3
    ).mean()

    result = pd.DataFrame({
        "Actual Sales": monthly,
        "3-Month Moving Average": forecast
    })

    print("\n========== SALES FORECAST ==========")
    print(result)

    return result


# --------------------------------------------------
# 14. BAR CHART
# --------------------------------------------------

def category_bar_chart(df):

    data = category_analysis(df)

    plt.figure(figsize=(8, 5))

    data.plot(kind="bar")

    plt.title("Sales by Product Category")
    plt.xlabel("Category")
    plt.ylabel("Sales")
    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        f"{CHART_FOLDER}/category_sales.png"
    )

    plt.show()


# --------------------------------------------------
# 15. LINE CHART
# --------------------------------------------------

def sales_line_chart(df):

    data = monthly_sales(df)

    plt.figure(figsize=(10, 5))

    plt.plot(
        data.index,
        data.values,
        marker="o"
    )

    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Sales")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        f"{CHART_FOLDER}/monthly_sales_trend.png"
    )

    plt.show()


# --------------------------------------------------
# 16. PIE CHART
# --------------------------------------------------

def category_pie_chart(df):

    data = category_analysis(df)

    plt.figure(figsize=(7, 7))

    plt.pie(
        data.values,
        labels=data.index,
        autopct="%1.1f%%"
    )

    plt.title("Sales Distribution by Category")

    plt.savefig(
        f"{CHART_FOLDER}/category_pie.png"
    )

    plt.show()


# --------------------------------------------------
# 17. REGION CHART
# --------------------------------------------------

def region_chart(df):

    data = region_analysis(df)

    plt.figure(figsize=(8, 5))

    data.plot(kind="bar")

    plt.title("Sales by Region")
    plt.xlabel("Region")
    plt.ylabel("Sales")

    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        f"{CHART_FOLDER}/regional_sales.png"
    )

    plt.show()


# --------------------------------------------------
# 18. DASHBOARD
# --------------------------------------------------

def create_dashboard(df):

    monthly = monthly_sales(df)
    category = category_analysis(df)
    region = region_analysis(df)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)

    plt.plot(
        monthly.index,
        monthly.values,
        marker="o"
    )

    plt.title("Monthly Sales")
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 2)

    category.plot(kind="bar")

    plt.title("Category Sales")

    plt.subplot(2, 2, 3)

    plt.pie(
        category.values,
        labels=category.index,
        autopct="%1.1f%%"
    )

    plt.title("Category Distribution")

    plt.subplot(2, 2, 4)

    region.plot(kind="bar")

    plt.title("Regional Sales")

    plt.tight_layout()

    plt.savefig(
        f"{CHART_FOLDER}/sales_dashboard.png"
    )

    plt.show()


# --------------------------------------------------
# 19. EXPORT TO EXCEL
# --------------------------------------------------

def export_excel(df):

    file_path = (
        f"{REPORT_FOLDER}/sales_analysis.xlsx"
    )

    with pd.ExcelWriter(
        file_path,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            sheet_name="Cleaned Data",
            index=False
        )

        category_analysis(df).to_excel(
            writer,
            sheet_name="Category Sales"
        )

        product_analysis(df).to_excel(
            writer,
            sheet_name="Product Sales"
        )

        monthly_sales(df).to_excel(
            writer,
            sheet_name="Monthly Sales"
        )

        region_analysis(df).to_excel(
            writer,
            sheet_name="Regional Sales"
        )

        customer_analysis(df).to_excel(
            writer,
            sheet_name="Customers"
        )

    print(
        f"\nExcel report saved to: {file_path}"
    )


# --------------------------------------------------
# 20. EXPORT CSV
# --------------------------------------------------

def export_csv(df):

    df.to_csv(
        "outputs/cleaned_sales.csv",
        index=False
    )

    monthly_sales(df).to_csv(
        f"{REPORT_FOLDER}/monthly_sales.csv"
    )

    category_analysis(df).to_csv(
        f"{REPORT_FOLDER}/category_sales.csv"
    )

    product_analysis(df).to_csv(
        f"{REPORT_FOLDER}/product_sales.csv"
    )

    print("\nCSV reports exported successfully.")


# --------------------------------------------------
# 21. EXECUTIVE SUMMARY
# --------------------------------------------------

def executive_summary(df):

    total_sales = df["Sales"].sum()

    avg_sale = df["Sales"].mean()

    top_product = (
        df.groupby("Product")["Sales"]
        .sum()
        .idxmax()
    )

    top_category = (
        df.groupby("Category")["Sales"]
        .sum()
        .idxmax()
    )

    top_region = (
        df.groupby("Region")["Sales"]
        .sum()
        .idxmax()
    )

    peak_date, peak_value = peak_sales(df)

    summary = f"""
SALES DATA ANALYSIS - EXECUTIVE SUMMARY
========================================

Total Sales:
₹{total_sales:,.2f}

Average Sale:
₹{avg_sale:,.2f}

Top Product:
{top_product}

Top Category:
{top_category}

Top Region:
{top_region}

Peak Sales Date:
{peak_date}

Peak Sales Value:
₹{peak_value:,.2f}

Total Orders:
{df["Order_ID"].nunique()}

Total Quantity Sold:
{df["Quantity"].sum()}
"""

    print(summary)

    with open(
        f"{REPORT_FOLDER}/executive_summary.txt",
        "w"
    ) as file:
        file.write(summary)

    print(
        "Executive summary saved successfully."
    )


# --------------------------------------------------
# 22. FULL ANALYSIS
# --------------------------------------------------

def full_analysis(df):

    print("\nRunning complete analysis...")

    basic_metrics(df)
    category_analysis(df)
    product_analysis(df)
    monthly_sales(df)
    monthly_growth(df)
    region_analysis(df)
    customer_analysis(df)
    customer_lifetime_value(df)
    peak_sales(df)
    sales_forecast(df)

    category_bar_chart(df)
    sales_line_chart(df)
    category_pie_chart(df)
    region_chart(df)
    create_dashboard(df)

    export_excel(df)
    export_csv(df)

    executive_summary(df)

    print("\n====================================")
    print("FULL ANALYSIS COMPLETED")
    print("====================================")


# --------------------------------------------------
# 23. COMMAND LINE INTERFACE
# --------------------------------------------------

def menu(df):

    while True:

        print("\n")
        print("====================================")
        print("      SALES DATA ANALYSIS SYSTEM")
        print("====================================")

        print("1. Explore Data")
        print("2. Clean Data")
        print("3. Basic Metrics")
        print("4. Category Analysis")
        print("5. Product Analysis")
        print("6. Monthly Sales")
        print("7. Monthly Growth")
        print("8. Customer Analysis")
        print("9. Regional Analysis")
        print("10. Sales Forecast")
        print("11. Peak Sales")
        print("12. Create Visualizations")
        print("13. Create Dashboard")
        print("14. Export Reports")
        print("15. Full Analysis")
        print("0. Exit")

        choice = input(
            "\nEnter your choice: "
        )

        if choice == "1":
            explore_data(df)

        elif choice == "2":
            df = clean_data(df)

        elif choice == "3":
            basic_metrics(df)

        elif choice == "4":
            category_analysis(df)

        elif choice == "5":
            product_analysis(df)

        elif choice == "6":
            monthly_sales(df)

        elif choice == "7":
            monthly_growth(df)

        elif choice == "8":
            customer_analysis(df)

        elif choice == "9":
            region_analysis(df)

        elif choice == "10":
            sales_forecast(df)

        elif choice == "11":
            peak_sales(df)

        elif choice == "12":
            category_bar_chart(df)
            sales_line_chart(df)
            category_pie_chart(df)
            region_chart(df)

        elif choice == "13":
            create_dashboard(df)

        elif choice == "14":
            export_excel(df)
            export_csv(df)
            executive_summary(df)

        elif choice == "15":
            full_analysis(df)

        elif choice == "0":
            print("\nThank you for using Sales Data Analysis System!")
            break

        else:
            print("\nInvalid choice. Please try again.")


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    df = load_data(DATA_FILE)

    if df is not None:

        df = clean_data(df)

        menu(df)