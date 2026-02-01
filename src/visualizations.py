import plotly.express as px


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

def plot_aisles_per_department(df):
    
    fig = px.bar(df, 
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

def plot_basket_size_distribution(df):

    mean = df['basket_size'].mean()
    median = df['basket_size'].median()

    fig = px.histogram(
        df, x='basket_size', nbins=50,
        title='Distribution of basket size',
        labels={'basket_size': 'Number of products'}
    )

    fig.add_vline(x=mean, line_color='red', line_dash='dash', line_width=2,
                  annotation_text=f'Mean: {mean:.1f}', annotation_position='top right')
    fig.add_vline(x=median, line_color='green', line_dash='dash', line_width=2,
                  annotation_text=f'Median: {median:.0f}', annotation_position='top left')

    fig.update_layout(yaxis_title='Number of orders')

    return fig

def plot_departments_per_basket(df):

    fig = px.histogram(
        df, x='nb_departments',
        title='Number of departments per basket',
        labels={'nb_departments': 'Number of departments'}
    )
    fig.update_layout(yaxis_title='Number of orders')
    return fig

def plot_basket_size_by_segment(df):

    fig = px.box(
        df, x='segment', y='basket_size',
        title='Basket size by customer segment',
        labels={'segment': 'Customer segment', 'basket_size': 'Basket size'},
        category_orders={'segment': ['Casual', 'Regular', 'Heavy']},
        color='segment'
    )

    fig.update_traces(showlegend=False)
    
    return fig

def plot_basket_size_vs_nb_orders(df):

    fig = px.scatter(
        df, x='nb_orders', y='avg_basket_size',
        title='Average basket size vs number of orders per user',
        labels={'nb_orders': 'Number of orders', 'avg_basket_size': 'Average basket size'},
        opacity=0.4
    )
    return fig