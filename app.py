"""
Streamlit Clustering + AutoViz App

A single-file Streamlit application for:
- Uploading CSV/XLSX files
- Selecting columns for clustering
- Running PyCaret clustering with user-selected algorithms
- Visualizing clusters
- Running AutoViz EDA colored by clusters
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
import io
import os
import tempfile
from pathlib import Path

# PyCaret imports
from pycaret.clustering import ClusteringExperiment

# AutoViz imports
from autoviz.AutoViz_Class import AutoViz_Class

# Plotting imports
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Page configuration
st.set_page_config(
    page_title="Clustering + AutoViz App",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_selected' not in st.session_state:
    st.session_state.df_selected = None
if 'df_clustered' not in st.session_state:
    st.session_state.df_clustered = None
if 'clustering_exp' not in st.session_state:
    st.session_state.clustering_exp = None
if 'selected_columns' not in st.session_state:
    st.session_state.selected_columns = []
if 'cluster_model' not in st.session_state:
    st.session_state.cluster_model = None


def load_data(file) -> Optional[pd.DataFrame]:
    """
    Load data from uploaded file (CSV or XLSX).
    
    Args:
        file: Uploaded file object
        
    Returns:
        DataFrame if successful, None otherwise
    """
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl')
        else:
            st.error("Unsupported file format. Please upload CSV or XLSX.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of numeric column names from dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        List of numeric column names
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_models_requiring_k() -> List[str]:
    """
    Get list of clustering model IDs that require num_clusters parameter.
    
    Returns:
        List of model IDs that support num_clusters
    """
    # Models that typically require/use num_clusters
    return [
        'kmeans',
        'kmodes',
        'ap',
        'meanshift',
        'sc',
        'hclust',
        'birch',
        'gmm'
    ]


def setup_clustering_experiment(df: pd.DataFrame) -> Optional[ClusteringExperiment]:
    """
    Setup PyCaret clustering experiment.
    
    Args:
        df: Dataframe with selected columns
        
    Returns:
        ClusteringExperiment instance or None if error
    """
    try:
        exp = ClusteringExperiment()
        exp.setup(data=df, session_id=42, silent=True, verbose=False)
        return exp
    except Exception as e:
        st.error(f"Error setting up clustering experiment: {str(e)}")
        return None


def get_available_models(exp: ClusteringExperiment) -> List[Tuple[str, str]]:
    """
    Get available clustering models from experiment.
    
    Args:
        exp: ClusteringExperiment instance
        
    Returns:
        List of tuples (model_id, model_name)
    """
    try:
        models_df = exp.models()
        return [(row['ID'], row['Name']) for _, row in models_df.iterrows()]
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        return []


def create_cluster_model(
    exp: ClusteringExperiment,
    model_id: str,
    num_clusters: Optional[int] = None
) -> Optional:
    """
    Create clustering model with optional num_clusters parameter.
    
    Args:
        exp: ClusteringExperiment instance
        model_id: Model identifier
        num_clusters: Number of clusters (optional)
        
    Returns:
        Trained model or None if error
    """
    try:
        if num_clusters is not None and model_id.lower() in get_models_requiring_k():
            model = exp.create_model(model_id, num_clusters=num_clusters)
        else:
            model = exp.create_model(model_id)
        return model
    except Exception as e:
        st.error(f"Error creating model: {str(e)}")
        return None


def assign_clusters(exp: ClusteringExperiment, model) -> Optional[pd.DataFrame]:
    """
    Assign cluster labels to dataframe.
    
    Args:
        exp: ClusteringExperiment instance
        model: Trained clustering model
        
    Returns:
        Dataframe with Cluster column added
    """
    try:
        assigned_df = exp.assign_model(model)
        return assigned_df
    except Exception as e:
        st.error(f"Error assigning clusters: {str(e)}")
        return None


def plot_clusters_2d(
    df: pd.DataFrame,
    method: str = 'pca'
) -> Optional:
    """
    Create 2D visualization of clusters using PCA or t-SNE.
    
    Args:
        df: Dataframe with Cluster column
        method: Dimensionality reduction method ('pca' or 'tsne')
        
    Returns:
        Plotly figure or None if error
    """
    try:
        # Get feature columns (exclude Cluster column)
        feature_cols = [col for col in df.columns if col != 'Cluster']
        X = df[feature_cols].values
        
        # Apply dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            X_reduced = reducer.fit_transform(X)
            x_label = 'PC1'
            y_label = 'PC2'
        else:  # tsne
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df)-1))
            X_reduced = reducer.fit_transform(X)
            x_label = 't-SNE 1'
            y_label = 't-SNE 2'
        
        # Create dataframe for plotting
        plot_df = pd.DataFrame({
            x_label: X_reduced[:, 0],
            y_label: X_reduced[:, 1],
            'Cluster': df['Cluster'].astype(str)
        })
        
        # Create plotly scatter plot
        fig = px.scatter(
            plot_df,
            x=x_label,
            y=y_label,
            color='Cluster',
            title=f'Cluster Visualization ({method.upper()})',
            hover_data={'Cluster': True}
        )
        fig.update_layout(height=600)
        return fig
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None


def run_autoviz(df: pd.DataFrame, output_dir: str) -> Optional[str]:
    """
    Run AutoViz EDA with Cluster as dependent variable.
    
    Args:
        df: Dataframe with Cluster column
        output_dir: Directory to save AutoViz output
        
    Returns:
        Filename of HTML file or None if error
    """
    try:
        av = AutoViz_Class()
        html_file = av.AutoViz(
            filename="",
            dfte=df,
            depVar='Cluster',
            verbose=0,
            lowess=False,
            chart_format='html',
            save_plot_dir=output_dir
        )
        # AutoViz returns filename, find the actual HTML file
        if html_file:
            # Check if it's a full path or just filename
            if os.path.exists(html_file):
                return os.path.basename(html_file) if os.path.dirname(html_file) == output_dir else html_file
            else:
                # Try to find HTML files in output directory
                html_files = list(Path(output_dir).glob("*.html"))
                if html_files:
                    return html_files[0].name
        return html_file
    except Exception as e:
        st.error(f"Error running AutoViz: {str(e)}")
        return None


def main():
    """Main application flow."""
    
    st.title("📊 Clustering + AutoViz App")
    st.markdown("Upload your data, choose columns, run clustering, and explore with AutoViz!")
    
    # Step 1: File Upload
    st.header("Step 1: Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV or XLSX file",
        type=['csv', 'xlsx'],
        help="Maximum file size: 50MB"
    )
    
    if uploaded_file is not None:
        # Check file size (50MB limit)
        if uploaded_file.size > 50 * 1024 * 1024:
            st.error("File size exceeds 50MB limit. Please upload a smaller file.")
            st.stop()
        
        # Load data
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state.df_original = df
            
            st.success(f"✅ File loaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Display preview
            with st.expander("Preview Data"):
                st.dataframe(df.head(10))
                st.caption(f"Total rows: {len(df)}")
            
            # Step 2: Column Selection
            st.header("Step 2: Select Columns for Clustering")
            
            numeric_cols = get_numeric_columns(df)
            if len(numeric_cols) == 0:
                st.warning("⚠️ No numeric columns found in the dataset. Clustering requires numeric features.")
                st.stop()
            
            selected_cols = st.multiselect(
                "Choose columns for clustering",
                options=df.columns.tolist(),
                default=numeric_cols,
                help="Select numeric columns to use for clustering"
            )
            
            if len(selected_cols) < 2:
                st.warning("⚠️ Please select at least 2 columns for clustering.")
                st.stop()
            
            st.session_state.selected_columns = selected_cols
            
            # Create subset dataframe
            df_selected = df[selected_cols].copy()
            
            # Check for missing values
            missing_count = df_selected.isnull().sum().sum()
            if missing_count > 0:
                st.warning(f"⚠️ Found {missing_count} missing values. They will be handled by PyCaret.")
                # Fill missing values with median for numeric columns
                df_selected = df_selected.fillna(df_selected.median())
            
            st.session_state.df_selected = df_selected
            
            # Display selected columns info
            with st.expander("Selected Columns Info"):
                st.dataframe(df_selected.describe())
            
            # Step 3: Clustering Configuration
            st.header("Step 3: Configure Clustering")
            
            if st.button("Initialize Clustering Experiment", type="primary"):
                with st.spinner("Setting up PyCaret clustering experiment..."):
                    exp = setup_clustering_experiment(df_selected)
                    if exp is not None:
                        st.session_state.clustering_exp = exp
                        st.success("✅ Clustering experiment initialized!")
            
            if st.session_state.clustering_exp is not None:
                # Get available models
                models = get_available_models(st.session_state.clustering_exp)
                
                if len(models) > 0:
                    # Create model selection dropdown
                    model_options = {f"{name} ({id})": id for id, name in models}
                    selected_model_display = st.selectbox(
                        "Choose Clustering Algorithm",
                        options=list(model_options.keys()),
                        help="Select a clustering algorithm from available PyCaret models"
                    )
                    selected_model_id = model_options[selected_model_display]
                    
                    # Check if model requires num_clusters
                    requires_k = selected_model_id.lower() in get_models_requiring_k()
                    
                    num_clusters = None
                    if requires_k:
                        num_clusters = st.number_input(
                            "Number of Clusters",
                            min_value=2,
                            max_value=min(20, len(df_selected) - 1),
                            value=3,
                            help="Specify the number of clusters to create"
                        )
                    else:
                        st.info(f"ℹ️ {selected_model_display} does not require specifying number of clusters.")
                    
                    # Step 4: Run Clustering
                    st.header("Step 4: Run Clustering")
                    
                    if st.button("Run Clustering", type="primary"):
                        with st.spinner("Running clustering algorithm..."):
                            model = create_cluster_model(
                                st.session_state.clustering_exp,
                                selected_model_id,
                                num_clusters
                            )
                            
                            if model is not None:
                                st.session_state.cluster_model = model
                                
                                # Assign clusters
                                df_clustered = assign_clusters(
                                    st.session_state.clustering_exp,
                                    model
                                )
                                
                                if df_clustered is not None:
                                    # Merge with original dataframe
                                    df_final = st.session_state.df_original.copy()
                                    df_final['Cluster'] = df_clustered['Cluster'].values
                                    st.session_state.df_clustered = df_final
                                    
                                    st.success("✅ Clustering completed!")
                                    
                                    # Display cluster summary
                                    st.subheader("Cluster Summary")
                                    cluster_counts = df_final['Cluster'].value_counts().sort_index()
                                    st.dataframe(cluster_counts.reset_index().rename(
                                        columns={'index': 'Cluster', 'Cluster': 'Count'}
                                    ))
                                    
                                    # Step 5: Visualize Clusters
                                    st.header("Step 5: Visualize Clusters")
                                    
                                    viz_method = st.radio(
                                        "Visualization Method",
                                        options=['pca', 'tsne'],
                                        horizontal=True,
                                        help="Choose dimensionality reduction method"
                                    )
                                    
                                    fig = plot_clusters_2d(df_clustered, method=viz_method)
                                    if fig is not None:
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Step 6: AutoViz EDA
                                    st.header("Step 6: AutoViz EDA")
                                    
                                    if st.button("Run AutoViz EDA", type="primary"):
                                        with st.spinner("Running AutoViz... This may take a few moments."):
                                            # Create temporary directory for AutoViz output
                                            with tempfile.TemporaryDirectory() as tmpdir:
                                                html_file = run_autoviz(df_final, tmpdir)
                                                
                                                if html_file:
                                                    # Try multiple possible paths
                                                    html_path = Path(tmpdir) / html_file
                                                    if not html_path.exists():
                                                        # Try finding any HTML file in the directory
                                                        html_files = list(Path(tmpdir).glob("*.html"))
                                                        if html_files:
                                                            html_path = html_files[0]
                                                    
                                                    if html_path.exists():
                                                        with open(html_path, 'r', encoding='utf-8') as f:
                                                            html_content = f.read()
                                                        
                                                        st.success("✅ AutoViz EDA completed!")
                                                        st.components.v1.html(html_content, height=800, scrolling=True)
                                                    else:
                                                        st.warning("AutoViz completed but HTML file not found. Check the output directory.")
                                                else:
                                                    st.warning("AutoViz did not generate output. Please check your data and try again.")
                                    
                                    # Step 7: Export Data
                                    st.header("Step 7: Export Clustered Data")
                                    
                                    csv = df_final.to_csv(index=False)
                                    st.download_button(
                                        label="Download Clustered Data as CSV",
                                        data=csv,
                                        file_name="clustered_data.csv",
                                        mime="text/csv",
                                        type="primary"
                                    )
                                    
                                    with st.expander("Preview Clustered Data"):
                                        st.dataframe(df_final.head(20))
                else:
                    st.error("No clustering models available.")
            else:
                st.info("👆 Please initialize the clustering experiment first.")
    else:
        st.info("👆 Please upload a CSV or XLSX file to get started.")


if __name__ == "__main__":
    main()

