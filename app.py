import streamlit as st
import pandas as pd
import plotly.express as px
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Config
st.set_page_config(page_title="Retail Dashboard", layout="wide", initial_sidebar_state="expanded")

# Load data
@st.cache_data
def load_data():
    base_path = "data/processed/"
    files = {
        "aisle": "aisle_performance.csv",
        "dept": "department_performance.csv",
        "rfm": "rfm_customer_segments.csv",
        "rules": "rules_clean.csv",
        "prod_list": "products_in_rules.csv" 
    }
    
    data = {}
    for key, name in files.items():
        full_path = os.path.join(base_path, name)
        if os.path.exists(full_path):
            data[key] = pd.read_csv(full_path)
        else:
            st.error(f"❌ Missing file: {name}")
            return None
    return data

data = load_data()

if data:
    df_aisle, df_dept, df_rfm = data['aisle'], data['dept'], data['rfm']
    df_rules, df_prod_list = data['rules'], data['prod_list']

    # Nav
    st.sidebar.header("Navigation")
    menu = st.sidebar.radio("Go to:", ["Overview"])

    # Overwiev
    if menu == "Overview":
        st.title("Dashboard: Performance Overview")

        # Segment Filter
        segments_disponibles = ["All Segments"] + list(df_rfm['segment'].unique())
        segment_choisi = st.selectbox("Filter indicators by segment:", segments_disponibles)

        if segment_choisi == "All Segments":
            df_filtered = df_rfm
        else:
            df_filtered = df_rfm[df_rfm['segment'] == segment_choisi]
        
        # KPI
        total_rev = df_filtered['total_spent_eur'].sum()
        total_ord = df_filtered['num_orders'].sum()
        total_itm = df_filtered['total_items'].sum()
        total_cust = len(df_filtered)
        
        aov = total_rev / total_ord if total_ord > 0 else 0
        items_per_order = total_itm / total_ord if total_ord > 0 else 0

        st.subheader(f"Statistics: {segment_choisi}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Revenue", f"{total_rev:,.2f} €")
        m2.metric("Total Orders", f"{total_ord:,}")
        m3.metric("Items Sold", f"{total_itm:,.0f}")
        m4.metric("Segment Customers", f"{total_cust:,}")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Avg Basket", f"{aov:.2f} €")
        m6.metric("Items by Order", f"{items_per_order:.1f}")
        m7.metric("Revenue by Customer", f"{(total_rev/total_cust if total_cust > 0 else 0):.2f} €")
        m8.metric("Avg RFM Score", f"{df_filtered['RFM_score'].mean():.1f}/15")

        st.markdown("---")
        col_left, col_right = st.columns(2)
        
        # Graph left
        with col_left:
            st.subheader("Top 5 Departments & Top 10 Aisles")
            
            # Préparation des données
            top_5_depts = df_dept.nlargest(5, 'items_sold')
            top_10_aisles = df_aisle.nlargest(10, 'items_sold')

            fig_combined = make_subplots(
                rows=2, cols=1,
                row_heights=[0.3, 0.7],
                specs=[[{"type": "treemap"}], [{"type": "treemap"}]],
                vertical_spacing=0.05,
                subplot_titles=("Top 5 Departments", "Top 10 Aisles")
            )

            # Top 5
            fig_combined.add_trace(
                go.Treemap(
                    labels=top_5_depts['department'],
                    parents=[""] * 5,
                    values=top_5_depts['items_sold'],
                    marker=dict(colorscale='Blues'),
                    textinfo="label+value",
                    hovertemplate='<b>%{label}</b><br>Items: %{value:,}<extra></extra>'
                ),
                row=1, col=1
            )

            # Top 10
            fig_combined.add_trace(
                go.Treemap(
                    labels=top_10_aisles['aisle'],
                    parents=[""] * 10,
                    values=top_10_aisles['items_sold'],
                    marker=dict(colorscale='Greys'), # Différenciation par couleur
                    textinfo="label+value",
                    hovertemplate='<b>%{label}</b><br>Items: %{value:,}<extra></extra>'
                ),
                row=2, col=1
            )

            fig_combined.update_layout(height=700, margin=dict(t=30, b=10, l=0, r=0))
            st.plotly_chart(fig_combined, use_container_width=True)

        # Graph right
        with col_right:
            st.subheader("Revenue % by Customer Segment")
            
            total_rev_all = df_rfm['total_spent_eur'].sum()
            concentration = df_rfm.groupby('segment').agg({'user_id': 'count', 'total_spent_eur': 'sum'}).reset_index()
            concentration['% Customers'] = (concentration['user_id'] / len(df_rfm)) * 100
            concentration['% Revenue'] = (concentration['total_spent_eur'] / total_rev_all) * 100

            df_plot = concentration.melt(id_vars='segment', value_vars=['% Customers', '% Revenue'], 
                                        var_name='Metric', value_name='Percentage')

            fig_money = px.bar(
                df_plot, x='segment', y='Percentage', color='Metric', barmode='group',
                color_discrete_map={'% Customers': '#94a3b8', '% Revenue': '#3b82f6'}
            )
            fig_money.update_layout(height=700) # Aligné sur la hauteur du bloc de gauche
            st.plotly_chart(fig_money, use_container_width=True)

else:
    st.warning("Please check your data files in 'data/processed/'.")