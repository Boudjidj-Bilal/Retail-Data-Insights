import plotly.express as px
import plotly.graph_objects as go

def plot_distribution_of_orders(df):

    mean_val = df['nb_orders'].mean()
    median_val = df['nb_orders'].median()

    fig= px.histogram(df, 
                      x='nb_orders',
                      nbins=50,
                      title='Distribution of number of orders per user'
    )

    fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {mean_val:.1f}")
    fig.add_vline(x=median_val, line_dash="dot", line_color="green",
                  annotation_text=f"Median: {median_val:.0f}")
    
    fig.update_layout(
        xaxis_title='Number of Orders',
        yaxis_title='Number of Users'
    )

    return fig

def plot_order_hours(df):
    
    fig = px.histogram(df, 
                       x='order_hour_of_day',
                       nbins=24,
                       title='Distribution of orders by hour of day'
    )
    
    fig.update_layout(
        xaxis_title='Hour of day',
        yaxis_title='Number of orders'
    )
    
    return fig


def plot_orders_by_dow(df):
    
    dow_distribution = df['order_dow'].value_counts().sort_index()

    day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 
                 'Thursday', 'Friday', 'Saturday']
    
    fig = px.bar(x=day_names, 
                 y=dow_distribution.values,
                 title='Distribution of orders by day of week')
    
    fig.update_traces(marker_color='coral')
    
    fig.update_layout(
        xaxis_title='Day of week',
        yaxis_title='Number of orders'
    )
    
    return fig

def plot_aisles_per_department(aisles_per_department):
    
    fig = px.bar(aisles_per_department, 
                 x='department',
                 y='nb_aisles',
                 title='Number of aisles per department',
                 color='nb_aisles',
                 color_continuous_scale='Greens')
    
    fig.update_layout(
        xaxis_title='Department',
        yaxis_title='Number of aisles',
        showlegend=False
    )
    
    return fig

def plot_sales_by_department_barplot(df):
    
    fig = px.bar(df, 
                 x='department',
                 y='total_sales',
                 title='Total sales per department',
                 color='total_sales',
                 color_continuous_scale='Reds')
    
    fig.update_layout(
        xaxis_title='Department',
        yaxis_title='Total sales',
        showlegend=False
    )
    
    return fig

def plot_sales_by_department_boxplot(df):

    fig = px.box(df,
                 x='department',
                 y='nb_sales',
                 log_y=True)            
    
    return fig

def plot_sales_by_aisle_barplot(df):

    fig = px.bar(df,
                 x='aisle',
                 y='total_sales',
                 title='Total sales by aisle (Top 20)',
                 color='total_sales',
                 color_continuous_scale='Oranges')
    
    fig.update_layout(
        xaxis_title='Aisle',
        yaxis_title='Total sales',
        showlegend=False,
    )
    
    return fig 

def plot_sales_by_aisle_boxplot(df):

    fig = px.box(df,
                 x='aisle',
                 y='nb_sales',
                 log_y=True)
    
    return fig

def plot_top_20_products_barplot(df):

    fig = px.bar(df, 
             x='nb_sales', 
             y='product_name',
             orientation='h',
             title='Top 20 most purchased products',
             labels={'nb_sales': 'Number of purchases', 'product_name': 'Product'},
             color='nb_sales',
             color_continuous_scale='Greens')

    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=600,
        showlegend=False
    )

    return fig 

def scatterplot_top_100_products(df):

    fig = px.scatter(df,
                 x='nb_sales',
                 y='reorder_rate',
                 size='nb_sales',
                 color='department',
                 hover_data=['product_name', 'aisle'],
                 title='Product popularity vs customer loayalty (Top 100)',
                 labels={'nb_sales': 'Total sales', 
                         'reorder_rate': 'Reorder rate'})

    fig.update_layout(
        yaxis_tickformat='.0%',
        height=600
    )

    return fig

