import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Path settings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def plot_interactive_metrics(all_results: dict):
    data = []
    for model_name, res in all_results.items():
        metrics = res.get("metrics", {})
        for m_name, m_val in metrics.items():
            data.append({
                "Model": model_name.replace("Regressor", "").replace("Regression", ""),
                "Metric": m_name,
                "Value": m_val
            })
    df = pd.DataFrame(data)
    fig = px.bar(df, x="Metric", y="Value", color="Model", barmode="group",
                 title="Model Performans Karşılaştırması",
                 template="plotly_dark")
    return fig

def plot_interactive_feature_importance(model_name: str, feature_importance: dict):
    if not feature_importance: return None
    df = pd.DataFrame({"Feature": list(feature_importance.keys()), "Importance": list(feature_importance.values())}).sort_values("Importance", ascending=True)
    fig = px.bar(df, x="Importance", y="Feature", orientation='h', title=f"Feature Importance — {model_name}", template="plotly_dark")
    return fig

def plot_interactive_time_series(df_pd, date_col, target_col):
    fig = px.line(df_pd, x=date_col, y=target_col, title=f"{target_col} Zaman Serisi Trendi", template="plotly_dark")
    fig.update_xaxes(rangeslider_visible=True)
    return fig

def plot_interactive_distribution(df_pd, col, plot_type="histogram"):
    if plot_type == "histogram":
        fig = px.histogram(df_pd, x=col, nbins=50, title=f"{col} Dağılımı", template="plotly_dark", marginal="box")
    else:
        counts = df_pd[col].value_counts().reset_index()
        counts.columns = [col, "count"]
        fig = px.pie(counts, values="count", names=col, title=f"{col} Dağılımı (Yüzde)", template="plotly_dark", hole=0.4)
    return fig

def plot_interactive_eda_scatter(df_pd, x_col, y_col):
    fig = px.scatter(df_pd, x=x_col, y=y_col, trendline="ols", title=f"{x_col} vs {y_col}", template="plotly_dark", opacity=0.6)
    return fig

def plot_interactive_confusion_matrix(y_true, y_pred, labels=None):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    z = cm.tolist()
    x = labels if labels else [f"Class {i}" for i in range(len(z))]
    fig = ff.create_annotated_heatmap(z, x=x, y=x, colorscale='Viridis')
    fig.update_layout(title="Confusion Matrix", template="plotly_dark")
    return fig

def plot_interactive_roc_curve(y_true, y_score):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig = px.area(x=fpr, y=tpr, title=f'ROC Curve (AUC={roc_auc:.4f})', labels=dict(x='FPR', y='TPR'), template="plotly_dark")
    fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    return fig

def plot_interactive_actual_vs_pred(actuals, predictions):
    fig = px.scatter(x=actuals, y=predictions, labels={'x': 'Gerçek', 'y': 'Tahmin'}, title="Gerçek vs Tahmin", template="plotly_dark", opacity=0.5)
    lims = [min(min(actuals), min(predictions)), max(max(actuals), max(predictions))]
    fig.add_shape(type='line', line=dict(dash='dash', color='red'), x0=lims[0], x1=lims[1], y0=lims[0], y1=lims[1])
    return fig

def plot_interactive_residuals(residuals):
    fig = px.histogram(x=residuals, nbins=50, title="Residual Dağılımı", template="plotly_dark", marginal="rug")
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    return fig

def plot_interactive_correlation(df):
    corr = df.select_dtypes(include=[np.number]).corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Özellik Korelasyon Matrisi", template="plotly_dark", color_continuous_scale='RdBu_r')
    return fig

def plot_interactive_missing_values(df):
    null_counts = df.isnull().sum().reset_index()
    null_counts.columns = ['Sütun', 'Eksik_Sayisi']
    fig = px.bar(null_counts, x='Sütun', y='Eksik_Sayisi', title="Eksik Değer Analizi", template="plotly_dark", color='Eksik_Sayisi', color_continuous_scale='Reds')
    return fig

def plot_interactive_hourly_traffic(df):
    if 'Date' not in df.columns: return None
    df['Hour'] = pd.to_datetime(df['Date']).dt.hour
    hourly = df.groupby('Hour').size().reset_index(name='Count')
    fig = px.line(hourly, x='Hour', y='Count', title="Saatlik İşlem Trafiği", template="plotly_dark", markers=True)
    return fig
