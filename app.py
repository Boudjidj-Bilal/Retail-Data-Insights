import streamlit as st
import pandas as pd
import plotly.express as px
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(page_title="Retail Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data():
    base_path = "data/processed/"
    files = {
        "aisle": "aisle_performance.csv",
        "dept": "department_performance.csv",
        "rfm": "rfm_customer_segments.csv",
        "impulse": "impulse_products.csv",
        "enriched": "products_enriched.csv",
        "prod_segment": "products_per_segment.csv"
    }
    
    data = {}
    for key, name in files.items():
        full_path = os.path.join(base_path, name)
        if os.path.exists(full_path):
            data[key] = pd.read_csv(full_path)
        else:
            st.error(f"Fichier manquant : {name}")
            return None
    return data

data = load_data()

if data:
    df_rfm = data['rfm']
    # Calcul du score RFM global s'il n'est pas déjà dans le fichier
    if 'RFM_score' not in df_rfm.columns:
        df_rfm['RFM_score'] = df_rfm['R_score'] + df_rfm['F_score'] + df_rfm['M_score']
    
    # --- NAVIGATION ---
    st.sidebar.header("Navigation")
    menu = st.sidebar.radio("Go to:", ["Overview", "Customer Segmentation"])

    # --- PAGE 1: OVERVIEW ---
    if menu == "Overview":
        st.title("Dashboard: Performance Overview")

        # Filtre Segments
        segments_list = ["All Segments"] + sorted(list(df_rfm['segment'].unique()))
        segment_choisi = st.selectbox("Filter indicators by segment:", segments_list)

        df_filtered = df_rfm if segment_choisi == "All Segments" else df_rfm[df_rfm['segment'] == segment_choisi]
        
        # --- CALCUL DES 8 KPIs ---
        total_rev = df_filtered['total_spent_eur'].sum()
        total_ord = df_filtered['num_orders'].sum()
        total_cust = len(df_filtered)
        
        aov = total_rev / total_ord if total_ord > 0 else 0
        rev_per_cust = total_rev / total_cust if total_cust > 0 else 0
        avg_days = df_filtered['avg_days_between_orders'].mean()
        avg_rfm = df_filtered['RFM_score'].mean()

        st.subheader(f"Statistics: {segment_choisi}")
        
        # Ligne 1 des KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Revenue", f"{total_rev:,.2f} €")
        m2.metric("Total Orders", f"{total_ord:,}")
        m3.metric("Unique Customers", f"{total_cust:,}")
        m4.metric("Revenue / Customer", f"{rev_per_cust:,.2f} €")

        # Ligne 2 des KPIs
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Avg Basket (AOV)", f"{aov:.2f} €")
        m6.metric("Avg Days Between Orders", f"{avg_days:.1f} days")
        m7.metric("Avg RFM Score", f"{avg_rfm:.1f} / 15")
        m8.metric("Active Segments", len(df_filtered['segment'].unique()))

        st.markdown("---")
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Top 5 Departments & Top 10 Aisles")
            top_5_depts = data['dept'].nlargest(5, 'items_sold')
            top_10_aisles = data['aisle'].nlargest(10, 'items_sold')

            fig_combined = make_subplots(
                rows=2, cols=1, row_heights=[0.3, 0.7],
                specs=[[{"type": "treemap"}], [{"type": "treemap"}]],
                vertical_spacing=0.07,
                subplot_titles=("Top 5 Departments (Global)", "Top 10 Aisles (Global)")
            )
            fig_combined.add_trace(go.Treemap(labels=top_5_depts['department'], parents=[""] * 5, values=top_5_depts['items_sold'], marker=dict(colorscale='Blues'), textinfo="label+value"), row=1, col=1)
            fig_combined.add_trace(go.Treemap(labels=top_10_aisles['aisle'], parents=[""] * 10, values=top_10_aisles['items_sold'], marker=dict(colorscale='Greys'), textinfo="label+value"), row=2, col=1)
            fig_combined.update_layout(height=700, margin=dict(t=30, b=10, l=0, r=0))
            st.plotly_chart(fig_combined, use_container_width=True)

        with col_right:
            st.subheader("Revenue Concentration")
            concentration = df_rfm.groupby('segment').agg({'user_id': 'count', 'total_spent_eur': 'sum'}).reset_index()
            concentration['% Customers'] = (concentration['user_id'] / len(df_rfm)) * 100
            concentration['% Revenue'] = (concentration['total_spent_eur'] / df_rfm['total_spent_eur'].sum()) * 100
            df_plot = concentration.melt(id_vars='segment', value_vars=['% Customers', '% Revenue'], var_name='Metric', value_name='Percentage')
            fig_money = px.bar(df_plot, x='segment', y='Percentage', color='Metric', barmode='group', color_discrete_map={'% Customers': '#94a3b8', '% Revenue': '#3b82f6'})
            fig_money.update_layout(height=700)
            st.plotly_chart(fig_money, use_container_width=True)

    # --- PAGE 2: CUSTOMER SEGMENTATION ---
    elif menu == "Customer Segmentation":
        st.title("Customer Segmentation & Behavioral Insights")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Segment Mix")
            fig_pie = px.pie(df_rfm, names='segment', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            st.subheader("Average RFM Profile")
            avg_rfm_scores = df_rfm.groupby('segment')[['R_score', 'F_score', 'M_score']].mean().reset_index()
            fig_bar_rfm = px.bar(avg_rfm_scores, x='segment', y=['R_score', 'F_score', 'M_score'], barmode='group')
            st.plotly_chart(fig_bar_rfm, use_container_width=True)

        st.markdown("---")

        st.subheader("Top 10 Favorite Products per Segment")
        if 'prod_segment' in data:
            df_ps = data['prod_segment']
            seg_to_show = st.selectbox("Select a segment to explore its favorites:", df_ps['segment'].unique())
            products_list = df_ps[df_ps['segment'] == seg_to_show]['products'].values[0]
            items = [item.strip() for item in products_list.split(',')]
            cols = st.columns(2)
            for i, item in enumerate(items):
                cols[i % 2].write(f"**{i+1}.** {item}")
        
        st.markdown("---")

        st.subheader("Behavioral Product Insights")
        t1, = st.tabs(["Top 5 Impulse Buys"])
        
        with t1:
            top_1_impulse = data['impulse'].nlargest(1, 'impulse_ratio').iloc[0]
            st.metric("Impulse Champion", top_1_impulse['product_name'], f"Ratio: {top_1_impulse['impulse_ratio']:.2f}")
            top_5_impulse = data['impulse'].nlargest(5, 'impulse_ratio')
            fig_imp = px.bar(top_5_impulse, x='impulse_ratio', y='product_name', orientation='h', color='impulse_ratio', color_continuous_scale='Reds')
            fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}, height=350)
            st.plotly_chart(fig_imp, use_container_width=True)

else:
    st.warning("Veuillez vérifier que les fichiers CSV sont bien présents dans 'data/processed/'.")