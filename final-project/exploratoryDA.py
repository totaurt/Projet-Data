import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
from fpdf import FPDF
import io
import sys

# Walmart's color palette
colors = {
    "Blue Ink": "#041E41",
    "Pink": "#e01884",
    "White": "#ffffff",
    "Walnut Blue": "#001c3e",
    "Spark Yellow": "#ffc220"
}
walmart_palette = [colors["Blue Ink"], colors["Pink"], colors["Walnut Blue"], colors["Spark Yellow"]]
sns.set_palette(walmart_palette)

# Function to save and display plots
def save_and_show_plot(plot_name):
    plt.savefig(os.path.join(output_dir, plot_name), bbox_inches='tight')
    print(f"Saved plot to {os.path.join(output_dir, plot_name)}")
    plt.show()

# Output directory for saving images
output_dir = "final-project/output_images"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Load preprocessed data
preprocessed_path = "final-project\data\preprocessed\preprocessed_data.csv\preprocessed_data.csv"
data = pd.read_csv(preprocessed_path)

# ---------------------------
# General Analysis
# ---------------------------

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    data[['actual_demand', 'forecasted_demand', 'stockout_indicator', 'promotion_applied', 'inventory_level']].corr(),
    annot=True, cmap="coolwarm"
)
plt.title('Feature Correlation Heatmap', color=colors["Blue Ink"])
heatmap_file = os.path.join(output_dir, "feature_correlation_heatmap.png")
plt.savefig(heatmap_file, format='png', dpi=300, bbox_inches='tight')
print(f"Saved plot to {heatmap_file}")
plt.show()

# Total Demand Over Time
data['transaction_date'] = pd.to_datetime(data['transaction_date'])
demand_trend = data.groupby(data['transaction_date'].dt.to_period('M')).actual_demand.sum().reset_index()
demand_trend['transaction_date'] = demand_trend['transaction_date'].astype(str)

plt.figure(figsize=(10, 6))
ax = sns.lineplot(data=demand_trend, x='transaction_date', y='actual_demand', color=colors["Spark Yellow"])
for x, y in zip(demand_trend['transaction_date'], demand_trend['actual_demand']):
    plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8, color=colors["Blue Ink"])
plt.title('Total Demand Over Time', color=colors["Blue Ink"])
plt.xlabel('Transaction Date', color=colors["Blue Ink"])
plt.ylabel('Total Demand', color=colors["Blue Ink"])
plt.xticks(rotation=45)
plt.tight_layout()
demand_file = os.path.join(output_dir, "total_demand_over_time.png")
plt.savefig(demand_file, format='png', dpi=300, bbox_inches='tight')
print(f"Saved plot to {demand_file}")
plt.show()

# Total Sales by Category
if 'category' in data.columns and 'quantity_sold' in data.columns:
    sales_by_category = data.groupby('category')['quantity_sold'].sum().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=sales_by_category, x='category', y='quantity_sold')
    plt.title("Total Sales by Category", color=colors["Blue Ink"])
    plt.xlabel("Category", color=colors["Blue Ink"])
    plt.ylabel("Total Quantity Sold", color=colors["Blue Ink"])
    save_and_show_plot("total_sales_by_category.png")
    plt.show()

# Category Count
if 'category' in data.columns:
    category_count = data['category'].count().reset_index()
    category_count.columns = ['category', 'count']
    plt.figure(figsize=(10, 6))
    sns.barplot(data=category_count, x='category', y='count')
    plt.title("Category Count", color=colors["Blue Ink"])
    plt.xlabel("Category", color=colors["Blue Ink"])
    plt.ylabel("Count", color=colors["Blue Ink"])
    save_and_show_plot("category_count.png")
    plt.show()

# Trend Analysis by Weekday
weekday_demand = data.groupby('weekday').actual_demand.mean().reset_index()
ax = sns.barplot(data=weekday_demand, x='weekday', y='actual_demand')
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f', padding=3, fontsize=10, color='black')
plt.title('Average Demand by Weekday', color=colors["Blue Ink"])
plt.xlabel('Weekday', color=colors["Blue Ink"])
plt.ylabel('Average Actual Demand', color=colors["Blue Ink"])
save_and_show_plot('average_demand_by_weekday.png')
plt.show()

# Holiday Impact Analysis
holiday_demand = data.groupby('holiday_indicator').actual_demand.mean().reset_index()
ax = sns.barplot(data=holiday_demand, x='holiday_indicator', y='actual_demand')
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f', padding=3, fontsize=10, color='black')
plt.title('Impact of Holidays on Demand', color=colors["Blue Ink"])
plt.xlabel('Holiday Indicator (True/False)', color=colors["Blue Ink"])
plt.ylabel('Average Actual Demand', color=colors["Blue Ink"])
save_and_show_plot('holiday_impact_on_demand.png')
plt.show()

# Weather Conditions Impact
weather_demand = data.groupby('weather_conditions').actual_demand.mean().reset_index()
ax = sns.barplot(data=weather_demand, x='weather_conditions', y='actual_demand')
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f', padding=3, fontsize=10, color='black')
plt.title('Impact of Weather Conditions on Demand', color=colors["Blue Ink"])
plt.xlabel('Weather Conditions', color=colors["Blue Ink"])
plt.ylabel('Average Actual Demand', color=colors["Blue Ink"])
plt.xticks(rotation=75)
save_and_show_plot('weather_conditions_impact.png')
plt.show()

# ---------------------------
# Store Analysis
# ---------------------------

# Store Analysis
# Store Location X Actual Demand
ax = sns.barplot(data=data, x='store_location', y='actual_demand', errorbar=None)
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f', padding=3, fontsize=10, color='black')
plt.title('Store Location vs. Actual Demand')
save_and_show_plot('store_location_vs_actual_demand.png')
plt.show()

# Store Location X Stockout Indicator
sns.barplot(data=data, x='store_location', y='stockout_indicator', errorbar= None)
plt.title('Store Location vs. Stockout Indicator')
save_and_show_plot('store_location_vs_stockout_indicator.png')
plt.show()

# Store Location X Inventory Level
ax = sns.barplot(data=data, x='store_location', y='inventory_level', errorbar=None)
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f', padding=3, fontsize=10, color='black')
plt.title('Store Location vs. Inventory Level')
save_and_show_plot('store_location_vs_inventory_level.png')
plt.show()

# Promotion Impact by Location
if 'promotion_applied' in data.columns and 'quantity_sold' in data.columns and 'store_location' in data.columns:
    promo_data = data.groupby(['store_location', 'promotion_applied'])['quantity_sold'].sum().reset_index()
    ax = sns.barplot(data=promo_data, x='store_location', y='quantity_sold', hue='promotion_applied', palette=[colors["Blue Ink"], colors["Pink"]], errorbar=None)
    
    # Add data labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', label_type='edge', fontsize=10, color=colors["Blue Ink"])
    
    plt.title('Impact of Promotions on Sales by Location', color=colors["Blue Ink"])
    plt.xlabel('Store Location', color=colors["Blue Ink"])
    plt.ylabel('Total Quantity Sold', color=colors["Blue Ink"])
    plt.legend(title='Promotion Applied', labels=['No Promotion', 'Promotion'], loc='upper right')
    plt.tight_layout()
    save_and_show_plot('Promotion Impact by Location.png')
    plt.show()
    
# Inventory Below Reorder by Store
if 'inventory_below_reorder' in data.columns:
    inventory_data = data.groupby('store_location')['inventory_below_reorder'].sum().reset_index()
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=inventory_data, x='store_location', y='inventory_below_reorder', color=colors["Walnut Blue"], errorbar=None)
    ax.bar_label(ax.containers[0])
    plt.title('Items Below Reorder Point by Store', color=colors["Blue Ink"])
    plt.xlabel('Store Location', color=colors["Blue Ink"])
    plt.ylabel('Items Below Reorder Point', color=colors["Blue Ink"])
    plt.tight_layout()
    inventory_file = os.path.join(output_dir, "inventory_below_reorder_by_store.png")
    plt.savefig(inventory_file, format='png', dpi=300, bbox_inches='tight')
    print(f"Saved plot to {inventory_file}")
    plt.show()
    
    
# Reorder Quantity Analysis by Store Location
store_reorder = data.groupby('store_location').reorder_quantity.mean().reset_index()
ax = sns.barplot(data=store_reorder, x='store_location', y='reorder_quantity')
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f', padding=3, fontsize=10, color='black')
plt.title('Reorder Quantity by Store Location', color=colors["Blue Ink"])
plt.xlabel('Store Location', color=colors["Blue Ink"])
plt.ylabel('Average Reorder Quantity', color=colors["Blue Ink"])
save_and_show_plot('reorder_quantity_by_store_location.png')
plt.show()

# Supplier Lead Time by Store Location
store_lead_time = data.groupby('store_location').supplier_lead_time.mean().reset_index()
ax = sns.barplot(data=store_lead_time, x='store_location', y='supplier_lead_time')
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f', padding=3, fontsize=10, color='black')
plt.title('Supplier Lead Time by Store Location', color=colors["Blue Ink"])
plt.xlabel('Store Location', color=colors["Blue Ink"])
plt.ylabel('Average Supplier Lead Time', color=colors["Blue Ink"])
save_and_show_plot('supplier_lead_time_by_store_location.png')
plt.show()

# Inventory Analysis

# Inventory Level X Actual Demand
sns.lineplot(data=data, x='inventory_level', y='actual_demand', color=colors["Pink"])
plt.title('Inventory Level vs. Actual Demand')
save_and_show_plot('inventory_level_vs_actual_demand.png')
plt.show()

# Promotion Analysis

# Promotion Applied X Stockout Indicator
sns.barplot(data=data, x='promotion_applied', y='stockout_indicator', errorbar=None)
plt.title('Promotion Applied vs. Stockout Indicator')
save_and_show_plot('promotion_applied_vs_stockout_indicator.png')
plt.show()
# Forecast Analysis

# Distribution of Forecast Error
sns.boxplot(x=(data['actual_demand'] - data['forecasted_demand']), color=colors["Pink"])
plt.title('Distribution of Forecast Error')
plt.xlabel('Forecast Error (Actual Demand - Forecasted Demand)')
plt.ylabel('Value')
save_and_show_plot('distribution_of_forecast_error.png')
plt.show()

# ---------------------------
# Product Analysis
# ---------------------------

# Total Sales by Product
product_sales = data.groupby('product_name').quantity_sold.sum().reset_index()
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=product_sales, x='product_name', y='quantity_sold', color=colors["Spark Yellow"], errorbar=None)
ax.bar_label(ax.containers[0])
plt.title('Total Sales by Product', color=colors["Blue Ink"])
plt.xlabel('Product Name', color=colors["Blue Ink"])
plt.ylabel('Quantity Sold', color=colors["Blue Ink"])
plt.xticks(rotation=90)
plt.tight_layout()
sales_file = os.path.join(output_dir, "total_sales_by_product.png")
plt.savefig(sales_file, format='png', dpi=300, bbox_inches='tight')
print(f"Saved plot to {sales_file}")
plt.show()

# Total Quantity Sold Per Product
total_quantity_sold_per_product = data.groupby(['product_name']).agg({'quantity_sold': 'sum'}).reset_index()
total_quantity_sold_per_product = total_quantity_sold_per_product.sort_values('quantity_sold')
ax = sns.barplot(data=total_quantity_sold_per_product, y='product_name', x='quantity_sold', errorbar=None)
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f', padding=3, fontsize=10, color='black')
plt.title('Total Quantity Sold Per Product')
plt.xticks(rotation=75)
save_and_show_plot('total_quantity_sold_per_product.png')
plt.show()

# Unit Price X Product Name
ax = sns.barplot(data=data.groupby('product_name').unit_price.mean().reset_index(), x='product_name', y='unit_price', errorbar=None)
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10, color='black')
plt.title('Unit Price by Product Name')
plt.xticks(rotation=75)
save_and_show_plot('unit_price_by_product_name.png')
plt.show()

# Reorder Point vs. Product Name
reorder_point_product = data.groupby('product_name').reorder_point.mean().reset_index()
ax = sns.barplot(data=reorder_point_product, y='product_name', x='reorder_point')
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f', padding=3, fontsize=10, color='black')
plt.title('Reorder Point by Product Name', color=colors["Blue Ink"])
plt.xlabel('Reorder Point', color=colors["Blue Ink"])
plt.ylabel('Product Name', color=colors["Blue Ink"])
plt.xticks(rotation=75)
save_and_show_plot('reorder_point_by_product_name.png')
plt.show()


# ---------------------------
# Customer Analysis
# ---------------------------
# Customer Gender Analysis
gender_demand = data.groupby('customer_gender').actual_demand.mean().reset_index()
ax = sns.barplot(data=gender_demand, x='customer_gender', y='actual_demand')
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f', padding=3, fontsize=10, color='black')
plt.title('Customer Gender Analysis', color=colors["Blue Ink"])
plt.xlabel('Customer Gender', color=colors["Blue Ink"])
plt.ylabel('Average Actual Demand', color=colors["Blue Ink"])
save_and_show_plot('customer_gender_analysis.png')
plt.show()

# Age Distribution of Customers
sns.histplot(data=data, x='customer_age', kde=True, color=colors["Pink"], bins=20)
plt.title('Age Distribution of Customers', color=colors["Blue Ink"])
plt.xlabel('Customer Age', color=colors["Blue Ink"])
plt.ylabel('Frequency', color=colors["Blue Ink"])
save_and_show_plot('age_distribution_of_customers.png')
plt.show()


# Customer Loyalty Analysis
if 'customer_loyalty_level' in data.columns and 'quantity_sold' in data.columns:
    loyalty_data = data.groupby('customer_loyalty_level').quantity_sold.sum().reset_index()
    ax = sns.barplot(data=loyalty_data, x='customer_loyalty_level', y='quantity_sold', errorbar=None)

    # Add data labels with adjusted position and formatting
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', label_type='edge', padding=8, color=colors["Blue Ink"], fontsize=12)

    plt.title('Sales by Customer Loyalty Level', color=colors["Blue Ink"], fontsize=16)
    plt.xlabel('Customer Loyalty Level', color=colors["Blue Ink"], fontsize=12)
    plt.ylabel('Quantity Sold', color=colors["Blue Ink"], fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()
    
# Customer Preferred Month to Purchase
customer_preferred_month = data.groupby(data['transaction_date'].dt.month).actual_demand.sum().reset_index()
ax = sns.barplot(data=customer_preferred_month, x='transaction_date', y='actual_demand', errorbar=None)
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f', padding=3, fontsize=10, color='black')
plt.title('Customer Preferred Month to Purchase')
save_and_show_plot('customer_preferred_month_to_purchase.png')
plt.show()

# Month of Offers
if 'promotion_applied' in data.columns:
    promo_month = data.groupby(data['transaction_date'].dt.month).promotion_applied.sum().reset_index()
    ax = sns.barplot(data=promo_month, x='transaction_date', y='promotion_applied', errorbar=None)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', padding=3, fontsize=10, color='black')
    plt.title('Month of Offers')
    save_and_show_plot('month_of_offers.png')
    plt.show()

# Rush Hour Analysis
rush_hour = data.groupby(data['transaction_date'].dt.hour).actual_demand.sum().reset_index()
sns.barplot(data=rush_hour, x='transaction_date', y='actual_demand', errorbar=None)
plt.title('Rush Hour Analysis')
plt.xlabel('Transaction Hour')
plt.ylabel('Actual Demand')
save_and_show_plot('rush_hour_analysis.png')
plt.show()

# Preferred Purchase Day
weekday_sales = data.groupby('weekday').transaction_id.count().reset_index()
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=weekday_sales, x='weekday', y='transaction_id', color=colors["Walnut Blue"], errorbar=None)
ax.bar_label(ax.containers[0])
plt.title('Preferred Purchase Day', color=colors["Blue Ink"])
plt.xlabel('Day of the Week', color=colors["Blue Ink"])
plt.ylabel('Number of Transactions', color=colors["Blue Ink"])
plt.tight_layout()
weekday_file = os.path.join(output_dir, "preferred_purchase_day.png")
plt.savefig(weekday_file, format='png', dpi=300, bbox_inches='tight')
print(f"Saved plot to {weekday_file}")
plt.show()

# Customer Income Category Analysis
data.loc[data['customer_income'] <= data['customer_income'].quantile(0.25), 'customer_income_category']= 'Forth'
data.loc[data['customer_income'].between(data['customer_income'].quantile(0.25),data['customer_income'].quantile(0.50)), 'customer_income_category']= 'Third'
data.loc[data['customer_income'].between(data['customer_income'].quantile(0.50),data['customer_income'].quantile(0.75)) , 'customer_income_category']= 'Second'
data.loc[data['customer_income'] >= data['customer_income'].quantile(0.75), 'customer_income_category']= 'First'

ax = sns.barplot(data=data.groupby(['customer_income_category']).quantity_sold.sum().reset_index(), x='customer_income_category', y='quantity_sold')
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f', padding=3, fontsize=10, color='black')
plt.title('Customer Income Category Analysis')
plt.xlabel('Customer Income Category')
plt.ylabel('Quantity Sold')
save_and_show_plot('customer_income_category_analysis.png')
plt.show()



