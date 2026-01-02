import pandas as pd
import numpy as np
from datetime import datetime

def load_and_preprocess_data():
    """
    Load and preprocess Olist datasets with advanced feature engineering
    Returns comprehensive vendor profiles with multiple performance metrics
    """
    # Load all datasets
    orders = pd.read_csv("data/olist_orders_dataset.csv")
    items = pd.read_csv("data/olist_order_items_dataset.csv")
    products = pd.read_csv("data/olist_products_dataset.csv")
    categories = pd.read_csv("data/product_category_name_translation.csv")
    reviews = pd.read_csv("data/olist_order_reviews_dataset.csv")
    payments = pd.read_csv("data/olist_order_payments_dataset.csv")
    customers = pd.read_csv("data/olist_customers_dataset.csv")
    sellers = pd.read_csv("data/olist_sellers_dataset.csv")
    
    # Convert date columns
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
    orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])
    
    # Filter only delivered orders for analysis
    orders_delivered = orders[orders["order_status"] == "delivered"].copy()
    
    # Calculate delivery performance
    orders_delivered['actual_delivery_days'] = (
        orders_delivered['order_delivered_customer_date'] - 
        orders_delivered['order_purchase_timestamp']
    ).dt.days
    
    orders_delivered['estimated_delivery_days'] = (
        orders_delivered['order_estimated_delivery_date'] - 
        orders_delivered['order_purchase_timestamp']
    ).dt.days
    
    orders_delivered['delivery_delay'] = (
        orders_delivered['actual_delivery_days'] - 
        orders_delivered['estimated_delivery_days']
    )
    
    # Merge all datasets
    df = items.merge(orders_delivered, on="order_id", how="inner")
    df = df.merge(products, on="product_id", how="left")
    df = df.merge(categories, on="product_category_name", how="left")
    df = df.merge(reviews, on="order_id", how="left")
    df = df.merge(payments, on="order_id", how="left")
    df = df.merge(sellers, on="seller_id", how="left")
    df = df.merge(customers, on="customer_id", how="left")
    
    # Feature Engineering
    df["revenue"] = df["price"] + df["freight_value"]
    df["profit_margin"] = df["price"] - (df["freight_value"] * 0.3)  # Estimate
    
    # Time-based features
    df['purchase_month'] = df['order_purchase_timestamp'].dt.to_period('M')
    df['purchase_quarter'] = df['order_purchase_timestamp'].dt.to_period('Q')
    
    # Aggregate to Vendor Profiles with ADVANCED metrics
    vendor_stats = df.groupby("seller_id").agg(
        # Revenue metrics
        total_revenue=("revenue", "sum"),
        avg_order_value=("revenue", "mean"),
        total_profit=("profit_margin", "sum"),
        
        # Volume metrics
        order_count=("order_id", "nunique"),
        total_items_sold=("order_item_id", "count"),
        unique_products=("product_id", "nunique"),
        unique_customers=("customer_id", "nunique"),
        
        # Performance metrics
        avg_review_score=("review_score", "mean"),
        review_count=("review_score", "count"),
        avg_delivery_days=("actual_delivery_days", "mean"),
        avg_delivery_delay=("delivery_delay", "mean"),
        on_time_delivery_rate=("delivery_delay", lambda x: (x <= 0).sum() / len(x) * 100),
        
        # Product metrics
        avg_price=("price", "mean"),
        avg_freight=("freight_value", "mean"),
        top_category=("product_category_name_english", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
        category_diversity=("product_category_name_english", "nunique"),
        
        # Payment metrics
        primary_payment_type=("payment_type", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
        
        # Geographic
        seller_state=("seller_state", "first"),
        seller_city=("seller_city", "first"),
        
        # Time metrics
        first_sale=("order_purchase_timestamp", "min"),
        last_sale=("order_purchase_timestamp", "max"),
        active_months=("purchase_month", "nunique")
    ).reset_index()
    
    # Calculate additional derived metrics
    vendor_stats['days_active'] = (vendor_stats['last_sale'] - vendor_stats['first_sale']).dt.days
    vendor_stats['sales_velocity'] = vendor_stats['order_count'] / (vendor_stats['days_active'] + 1)
    vendor_stats['revenue_per_product'] = vendor_stats['total_revenue'] / vendor_stats['unique_products']
    vendor_stats['customer_retention'] = vendor_stats['order_count'] / vendor_stats['unique_customers']
    vendor_stats['items_per_order'] = vendor_stats['total_items_sold'] / vendor_stats['order_count']
    
    # Performance scoring (0-100)
    vendor_stats['revenue_score'] = min_max_scale(vendor_stats['total_revenue'])
    vendor_stats['review_score'] = (vendor_stats['avg_review_score'] / 5) * 100
    vendor_stats['delivery_score'] = 100 - min_max_scale(vendor_stats['avg_delivery_delay'].clip(lower=0))
    vendor_stats['volume_score'] = min_max_scale(vendor_stats['order_count'])
    
    # Overall performance score (weighted average)
    vendor_stats['performance_score'] = (
        vendor_stats['revenue_score'] * 0.30 +
        vendor_stats['review_score'] * 0.30 +
        vendor_stats['delivery_score'] * 0.20 +
        vendor_stats['volume_score'] * 0.20
    )
    
    # Assign performance tiers
    vendor_stats['tier'] = pd.cut(
        vendor_stats['performance_score'],
        bins=[0, 40, 70, 100],
        labels=['C', 'B', 'A']
    )
    
    # Fill NaN values
    vendor_stats = vendor_stats.fillna({
        'avg_review_score': 3.0,
        'review_count': 0,
        'avg_delivery_delay': 0,
        'on_time_delivery_rate': 50
    })
    
    return vendor_stats

def min_max_scale(series):
    """Min-Max scaling to 0-100 range"""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([50] * len(series))
    return ((series - min_val) / (max_val - min_val)) * 100

def get_vendor_summary(vendor_stats, seller_id):
    """Get detailed summary for a specific vendor"""
    vendor = vendor_stats[vendor_stats['seller_id'] == seller_id].iloc[0]
    
    summary = {
        'seller_id': vendor['seller_id'],
        'tier': vendor['tier'],
        'performance_score': round(vendor['performance_score'], 2),
        
        'financial': {
            'total_revenue': round(vendor['total_revenue'], 2),
            'total_profit': round(vendor['total_profit'], 2),
            'avg_order_value': round(vendor['avg_order_value'], 2),
            'revenue_per_product': round(vendor['revenue_per_product'], 2)
        },
        
        'operations': {
            'order_count': int(vendor['order_count']),
            'total_items_sold': int(vendor['total_items_sold']),
            'items_per_order': round(vendor['items_per_order'], 2),
            'sales_velocity': round(vendor['sales_velocity'], 4)
        },
        
        'performance': {
            'avg_review_score': round(vendor['avg_review_score'], 2),
            'review_count': int(vendor['review_count']),
            'avg_delivery_days': round(vendor['avg_delivery_days'], 1),
            'on_time_delivery_rate': round(vendor['on_time_delivery_rate'], 1),
            'delivery_delay_avg': round(vendor['avg_delivery_delay'], 1)
        },
        
        'catalog': {
            'unique_products': int(vendor['unique_products']),
            'top_category': vendor['top_category'],
            'category_diversity': int(vendor['category_diversity']),
            'avg_price': round(vendor['avg_price'], 2)
        },
        
        'customer': {
            'unique_customers': int(vendor['unique_customers']),
            'customer_retention': round(vendor['customer_retention'], 2),
            'primary_payment': vendor['primary_payment_type']
        },
        
        'geographic': {
            'state': vendor['seller_state'],
            'city': vendor['seller_city']
        },
        
        'timeline': {
            'days_active': int(vendor['days_active']),
            'active_months': int(vendor['active_months']),
            'first_sale': str(vendor['first_sale'].date()),
            'last_sale': str(vendor['last_sale'].date())
        }
    }
    
    return summary