import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Config
st.set_page_config(page_title="Retail Dashboard", layout="wide", initial_sidebar_state="expanded")

# Chargement des datas
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
            st.error(f"❌ Fichier manquant : {name}")
            return None
    return data

data = load_data()

if data:
    df_aisle, df_dept, df_rfm = data['aisle'], data['dept'], data['rfm']
    df_rules, df_prod_list = data['rules'], data['prod_list']

    # Nav
    st.sidebar.header("Navigation")
    menu = st.sidebar.radio("Aller vers :", ["Vue d'ensemble"])

# Page 1
    if menu == "Vue d'ensemble":
        st.title("Tableau de Bord : Performance & Concentration")

        # Filtre segments
        segments_disponibles = ["Tous les segments"] + list(df_rfm['segment'].unique())
        segment_choisi = st.selectbox("Filtrer les indicateurs par segment :", segments_disponibles)

        if segment_choisi == "Tous les segments":
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

        st.subheader(f"Statistiques : {segment_choisi}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Chiffre d'Affaires", f"{total_rev:,.2f} €")
        m2.metric("Total Commandes", f"{total_ord:,}")
        m3.metric("Articles Vendus", f"{total_itm:,.0f}")
        m4.metric("Clients dans le Segment", f"{total_cust:,}")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Panier Moyen", f"{aov:.2f} €")
        m6.metric("Articles / Panier", f"{items_per_order:.1f}")
        m7.metric("CA / Client", f"{(total_rev/total_cust if total_cust > 0 else 0):.2f} €")
        m8.metric("Score RFM Moyen", f"{df_filtered['RFM_score'].mean():.1f}/15")

        st.markdown("---")
        col_left, col_right = st.columns(2)
        
        # Graphique 1 
        with col_left:
            st.subheader("Volume par Départements")
            fig_tree = px.treemap(
                df_dept, path=['department'], values='items_sold',
                color='items_sold', color_continuous_scale='Blues',
                labels={'department': 'Département', 'items_sold': 'Articles'}
            )
            fig_tree.update_traces(hovertemplate='<b>%{label}</b><br>Articles: %{value:,}')
            st.plotly_chart(fig_tree, use_container_width=True)
            

        # Graphique 2
        with col_right:
            st.subheader("% CA par segment client")
            
            total_cust_all = len(df_rfm)
            total_rev_all = df_rfm['total_spent_eur'].sum()

            concentration = df_rfm.groupby('segment').agg({
                'user_id': 'count',
                'total_spent_eur': 'sum'
            }).reset_index()

            concentration['% Clients'] = (concentration['user_id'] / total_cust_all) * 100
            concentration['% Chiffre d\'Affaires'] = (concentration['total_spent_eur'] / total_rev_all) * 100

            df_plot = concentration.melt(id_vars='segment', value_vars=['% Clients', '% Chiffre d\'Affaires'], 
                                        var_name='Métrique', value_name='Pourcentage')

            # 2. Crea du graphique
            fig_money = px.bar(
                df_plot, 
                x='segment', 
                y='Pourcentage', 
                color='Métrique', 
                barmode='group',
                color_discrete_map={'% Clients': '#94a3b8', '% Chiffre d\'Affaires': '#3b82f6'},
                labels={'segment': 'Segment Client', 'Pourcentage': 'Part du Total (%)'},
            )
            st.plotly_chart(fig_money, use_container_width=True)