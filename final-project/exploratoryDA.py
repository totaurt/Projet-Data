import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


raw_data_path = "final-project/data/raw/Walmart.csv"
data = pd.read_csv(raw_data_path)

print(data.head())
print(data.info())
print(data.describe())
print(data.sample())

# Catogories count
if 'category' in data.columns:
    print(data['category'].value_counts())
    sns.countplot(data['category'])
    plt.title("Catogories count")
    plt.show()

sales_by_category = data.groupby('product_name')['quantity_sold'].sum()
sales_by_category.plot(kind='bar', title="Total Sales by Category")
plt.title('Sales by Category')
plt.show()

sns.heatmap(data[['actual_demand','forecasted_demand','stockout_indicator',
                  'promotion_applied','inventory_level']].corr(),annot=True)
plt.title('Heatmap')
plt.show()

fig, ax1 = plt.subplots(figsize=(8, 6))

grouped = data.groupby('store_location')[['actual_demand', 'forecasted_demand']].mean()
grouped.plot(kind='bar', ax=ax1, color=['black', 'green'], width=0.8)
ax1.set_ylabel('Demand')
ax1.set_xlabel('Store Location')
plt.xticks(rotation=75)
plt.title('Actual Demand vs Forecasted Demand')
plt.tight_layout()
plt.show()

each_store_stockout = data.groupby('store_location').agg({'stockout_indicator':'sum'}) .reset_index()
sns.barplot(each_store_stockout, y = 'store_location', x='stockout_indicator')
plt.title('Store Stockout')
plt.show()

each_store_stockout = data.groupby(['store_location','promotion_applied']).agg({'quantity_sold':'sum'}) .reset_index()
sns.barplot(each_store_stockout, y = 'store_location', x='quantity_sold',hue='promotion_applied')
plt.title('Store Stockout with Promotion Applied')
plt.show()

sns.barplot(data.groupby('store_location').inventory_level.sum().reset_index(), y = 'store_location', x = 'inventory_level')
plt.title('Inventory Level')
plt.show()

total_quantity_sold_per_product = data.groupby(['product_name', 'promotion_applied']).agg({'quantity_sold': 'sum'}).reset_index()

total_quantity_sold_per_product = total_quantity_sold_per_product.sort_values('quantity_sold', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(data=total_quantity_sold_per_product, x='product_name', y='quantity_sold', hue='promotion_applied', dodge=True)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('Product Name', fontsize=12)
plt.ylabel('Total Quantity Sold', fontsize=12)
plt.title('Total Quantity Sold for Each Product', fontsize=14)
plt.legend(title='Promotion Applied', loc='upper right', fontsize=10)
plt.tight_layout()
plt.show()

sns.lineplot(data, y= 'unit_price', x  = 'product_name',errorbar= None)
plt.xticks(rotation= 75)
plt.title('Unit Price by Product')
plt.show()

data['transaction_date'] = pd.to_datetime(data['transaction_date'], errors='coerce')

sold_units_per_month_product = data.groupby([data['transaction_date'].dt.month,'product_name',
                                             'promotion_applied']).agg({'quantity_sold':'sum'}).reset_index()

product_subset = data['product_name'].value_counts().index

fig, axes = plt.subplots(4, 2, figsize=(8,8))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot data with seaborn using hue in each subplot
for i, ax in  enumerate(axes):
    subset = sold_units_per_month_product[sold_units_per_month_product['product_name'] == product_subset[i]]
    sns.lineplot(subset, x='transaction_date', y='quantity_sold', ax=ax, palette='viridis', errorbar=None)
    ax.set_title(f'{product_subset[i]} Quantity Sold Trend Over months')
    ax.set_xlabel('transaction month')
    ax.set_ylabel('quantity sold')
plt.tight_layout()

sns.barplot(data.groupby(['weekday']).transaction_id.count().reset_index() , y = 'weekday',x='transaction_id')
plt.xlabel('number of transactions')
plt.title("customer favourite day to make a purchase")
plt.show()

sold_units_per_month = data.groupby([data['transaction_date'].dt.month]).agg({'quantity_sold':'sum'}).reset_index()
sns.lineplot(sold_units_per_month , x= 'transaction_date'  ,y ='quantity_sold', errorbar=None)
plt.xlabel('transaction_month')
plt.title("Customer Perefered Month to Puchase")
plt.show()


    



    


