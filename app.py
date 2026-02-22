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
        "prod_list": "products_in_rules.csv",
        "impulse": "impulse_products.csv",
        "enriched": "products_enriched.csv",
        "rules_by_department": "rules_by_department.csv"
    }
    
    data = {}
    for key, name in files.items():
        full_path = os.path.join(base_path, name)
        if os.path.exists(full_path):
            data[key] = pd.read_csv(full_path)
        else:
            st.error(f"Missing file: {name}")
            return None
    return data

data = load_data()

if data:
    df_aisle, df_dept, df_rfm = data['aisle'], data['dept'], data['rfm']
    df_rules, df_prod_list = data['rules'], data['prod_list']

    # Nav
    st.sidebar.header("Navigation")
    menu = st.sidebar.radio("Go to:", ["Overview", "Customer Segmentation", "Bundles & Simulation"])

    # Page 1
    if menu == "Overview":
        st.title("Dashboard: Performance Overview")

        # Segment Filter
        segments_disponibles = ["All Segments"] + list(df_rfm['segment'].unique())
        segment_choisi = st.selectbox("Filter indicators by segment:", segments_disponibles)

        df_filtered = df_rfm if segment_choisi == "All Segments" else df_rfm[df_rfm['segment'] == segment_choisi]
        
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
        
        with col_left:
            # Graph top 5 departments and top 10 aisles
            st.subheader("Top 5 Departments & Top 10 Aisles")
            top_5_depts = df_dept.nlargest(5, 'items_sold')
            top_10_aisles = df_aisle.nlargest(10, 'items_sold')

            fig_combined = make_subplots(
                rows=2, cols=1, row_heights=[0.3, 0.7],
                specs=[[{"type": "treemap"}], [{"type": "treemap"}]],
                vertical_spacing=0.05,
                subplot_titles=("Top 5 Departments", "Top 10 Aisles")
            )
            fig_combined.add_trace(go.Treemap(labels=top_5_depts['department'], parents=[""] * 5, values=top_5_depts['items_sold'], marker=dict(colorscale='Blues'), textinfo="label+value"), row=1, col=1)
            fig_combined.add_trace(go.Treemap(labels=top_10_aisles['aisle'], parents=[""] * 10, values=top_10_aisles['items_sold'], marker=dict(colorscale='Greys'), textinfo="label+value"), row=2, col=1)
            fig_combined.update_layout(height=700, margin=dict(t=30, b=10, l=0, r=0))
            st.plotly_chart(fig_combined, use_container_width=True)

        with col_right:
            #Graph revenue by segment
            st.subheader("Revenue % by Customer Segment")
            concentration = df_rfm.groupby('segment').agg({'user_id': 'count', 'total_spent_eur': 'sum'}).reset_index()
            concentration['% Customers'] = (concentration['user_id'] / len(df_rfm)) * 100
            concentration['% Revenue'] = (concentration['total_spent_eur'] / df_rfm['total_spent_eur'].sum()) * 100
            df_plot = concentration.melt(id_vars='segment', value_vars=['% Customers', '% Revenue'], var_name='Metric', value_name='Percentage')
            fig_money = px.bar(df_plot, x='segment', y='Percentage', color='Metric', barmode='group', color_discrete_map={'% Customers': '#94a3b8', '% Revenue': '#3b82f6'})
            fig_money.update_layout(height=700)
            st.plotly_chart(fig_money, use_container_width=True)

    # Page 2
    #Characteristics of customer segments
    elif menu == "Customer Segmentation":
        st.title("Customer Segmentation & Behavioral Insights")
        strat_data = {
            "Segment": ["Champions", "Loyal", "Potential Loyalist", "At Risk", "Hibernating"],
            "Characteristics": [
                "High value, high frequency, recent.",
                "Steady customers, high frequency.",
                "Recent spenders with growth potential.",
                "High spenders who haven't returned recently.",
                "Low frequency, low monetary, long time ago."
            ]
        }
        st.table(pd.DataFrame(strat_data))
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            # Graph customer segment distribution
            st.subheader("Customer Base Mix")
            fig_pie = px.pie(df_rfm, names='segment', hole=0.4, title="Proportion of Total Customers")
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            # Graph average RFM scores
            st.subheader("RFM Scores Distribution")
            avg_rfm = df_rfm.groupby('segment')[['R_score', 'F_score', 'M_score']].mean().reset_index()
            fig_bar_rfm = px.bar(avg_rfm, x='segment', y=['R_score', 'F_score', 'M_score'], barmode='group')
            st.plotly_chart(fig_bar_rfm, use_container_width=True)

        if 'impulse' in data:
            # Graph top 5 impulse products
            st.markdown("---")
            st.subheader("Impulse Buy Analysis")
            top_impulse = data['impulse'].nlargest(10, 'impulse_ratio')
            fig_impulse = px.bar(top_impulse, x='impulse_ratio', y='product_name', orientation='h', color='impulse_ratio', color_continuous_scale='Reds')
            st.plotly_chart(fig_impulse, use_container_width=True)

        if 'enriched' in data:
            # Graph top 5 products by reorder rate
            st.markdown("---")
            st.subheader("Product Loyalty & Reorder Rates")
            top_reorder = data['enriched'][data['enriched']['nb_sales'] > 1000].nlargest(5, 'reorder_rate')
            fig_reorder = px.scatter(top_reorder, x='nb_sales', y='reorder_rate', size='purchase_frequency', color='department', text='product_name')
            fig_reorder.update_traces(textposition='top center')
            st.plotly_chart(fig_reorder, use_container_width=True)

    # Page 3
    elif menu == "Bundles & Simulation":
        st.title("Smart Bundles & Revenue Projection")
        available_products = sorted(df_prod_list['product_name'].unique())
        base_product = st.selectbox("Select a base product:", available_products)

        paires = df_rules[df_rules['antecedent'].str.contains(base_product, na=False, case=False)]
        df_rules_dept = data['rules_by_department']
        trios = df_rules_dept[df_rules_dept['antecedent'].str.contains(base_product, na=False, case=False)]
        trios = trios[trios['antecedent'].str.contains(',', na=False)]

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.markdown("#### Best 2-Item Bundles (Pairs)")
            if not paires.empty:
                for _, row in paires.nlargest(5, 'lift').iterrows():
                    with st.expander(f"➕ {row['consequent']}"):
                        st.write(f"Confidence: {row['confidence']:.1%}, Lift: {row['lift']:.2f}")
            else: st.write("No strong pair found.")

        with col_b2:
            st.markdown("#### Best 3-Item Bundles (Trios)")
            if not trios.empty:
                for _, row in trios.nlargest(5, 'lift').iterrows():
                    with st.expander(f"🍱 {row['antecedent']} ➔ {row['consequent']}"):
                        st.write(f"Confidence: {row['confidence']:.1%}")
            else: st.write("No complex bundle found.")

        st.markdown("---")
        st.subheader("Potential Revenue Simulator")
        target_segment = st.selectbox("Target Segment:", df_rfm['segment'].unique())
        conv_rate = st.slider("Conversion Rate (%)", 0.5, 20.0, 5.0) / 100
        bundle_val = st.number_input("Average Bundle Price (€)", value=15.0)
        num_cust = len(df_rfm[df_rfm['segment'] == target_segment])
        est_rev = (num_cust * conv_rate) * bundle_val
        st.metric("Estimated Revenue", f"{est_rev:,.2f} €")

else:
    st.warning("Please check your data files in 'data/processed/'.")