import math
import numpy as np
import pandas as pd
import unicodedata

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# config seaborn
palette = "BrBG"
palette2 = "BuGn"
sns.set(style="ticks")


# Remove accent
def remove_acentuacao(palavra):

    # Normaliza a palavra para a forma NFD (separa os caracteres das marcas diacríticas)
    nfkd = unicodedata.normalize('NFD', palavra)
    
    # Filtra apenas os caracteres que não são marcas diacríticas (exclui acentos)
    return ''.join([c for c in nfkd if not unicodedata.combining(c)])


# k_sturges
def k_sturges(serie):
    k = int(1 + (10/3) * np.log10(len(serie)))
    
    return k


# Bucketing
def df_bins_bucket(df, col, len_bins):

    # Calculate min ad max values
    min_value = int(df[col].min())
    max_value = int(df[col].max())

    # Calculate range bins and labels
    bins = list(np.arange(min_value, max_value + len_bins, len_bins))
    label_bins = [f"{bins[i]}-{bins[i+1] - 1}" for i in range(len(bins) - 1)]
    
    # Create buckets
    bck = pd.cut(x=df[col], bins=bins, labels=label_bins, include_lowest=True)
    
    return bck


# Tabela de frequência relativa
def numeric_frequency_table(df, col, len_bins):

    # Calculate range bins and labels
    min_value, max_value = int(df[col].min()), int(df[col].max())
    bins = list(np.arange(min_value, max_value + len_bins, len_bins))
    label_bins = [f"{bins[i]}-{bins[i+1] - 1}" for i in range(len(bins) - 1)]

    # Create table
    percent = pd.value_counts(
        pd.cut(
            x=df[col],
            bins=bins,
            labels=label_bins,
            include_lowest=True
        ), normalize=True
    )
    
    # Create table
    frequency = pd.value_counts(
        pd.cut(
            x=df[col],
            bins=bins,
            labels=label_bins,
            include_lowest=True
        ), normalize=False
    )
    
    # Create Dataframe
    freq_table = pd.DataFrame(
            {'Frequency': frequency, 'Percent': percent}
            )
    
    
    return freq_table


# Plotar box plot
def plot_box_plot(df): 
    
    # filter df by numeric features
    df_plot = df.select_dtypes(exclude=[object, 'category'])
    
    # Definir subplot
    cols = 4
    rows = math.ceil(len(df_plot.columns) / cols)
    
    # Criar a figura e os subplots
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(14, 5 * rows)) 
    
    # Flattening os eixos para facilitar o loop
    axes = axes.flatten()
    
    # Loop para criar um boxplot para cada variável numérica
    for i, col in enumerate(df_plot.columns):
        sns.boxplot(data=df_plot, y=col, ax=axes[i], orient='V', palette='Blues')
        axes[i].set_title(col, fontsize=16)
        
     # Remover os subplots vazios (caso o número de variáveis seja menor que o número de subplots)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
   
    
    # Layout
    plt.suptitle('Box Plot - Numeric Features\n', fontsize=22)
    plt.tight_layout()
    
    plt.show()


# Função para plotar histogramas
def plot_hist_plot(df): 
    # Filtrar apenas as colunas numéricas
    df_plot = df.select_dtypes(exclude=[object, 'category'])
    
    # Definir o número de colunas e linhas para os subplots
    cols = 2
    rows = math.ceil(len(df_plot.columns) / cols)
    
    # Definir o tamanho da figura
    plt.figure(figsize=(10, 4 * rows))
    
    # Loop para criar um histograma para cada variável
    for i, col in enumerate(df_plot.columns):
        plt.subplot(rows, cols, i + 1)
        sns.histplot(df, x=col, color='darkslategray', kde=False)
        plt.title(f'{col}')
        plt.xlabel(None)
        
        # axis
        plt.gca().spines[['top', 'right']].set_visible(False)
    
    # Ajustar o layout
    plt.suptitle('Histogram - Numeric Features', fontsize=22)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# Visualizar correlações com a variável target utilizando boxplot
def box_plot_correlations_x(df, target): 
    
    # filter df by numeric features
    df_plot = df.select_dtypes(include=[object, 'category'])

    if len(df.select_dtypes(include=[object, 'category']).columns) == 0:
        return "Não existem variáveis categóricas no dataset"

    elif df[target].dtype == object or df[target].dtype == 'category':
        return "A variável target é categórica"

    else:

        # Definir subplot
        cols = 1
        rows = math.ceil(len(df_plot.columns) / cols)
        
        # Criar a figura e os subplots
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(8, 3.25 * rows)) 
        
        # Flattening os eixos para facilitar o loop
        axes = axes.flatten()
        
        # Loop para criar um boxplot para cada variável numérica
        for i, col in enumerate(df_plot.columns):
            sns.boxplot(data=df_plot, x=df_plot[col], y=df[target], ax=axes[i], orient='v', palette='Blues')
            axes[i].set_title(col, fontsize=14)
            
            # Remover os rótulos do eixo x
            axes[i].set(xlabel=None)  
            
         # Remover os subplots vazios (caso o número de variáveis seja menor que o número de subplots)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        # Layout
        plt.suptitle('Categorical Features vs Target\n', fontsize=18)
        plt.tight_layout()
        
        return plt.show()


# Visualizar correlações com a variável target utilizando boxplot
def box_plot_correlations(df, target, order_target=None): 
    
    # filter df by numeric features
    if df[target].dtype == object or df[target].dtype == 'category':
        df_plot = df.select_dtypes(exclude=[object, 'category'])
    else:
        df_plot = df.select_dtypes(include=[object, 'category'])

    if len(df_plot.columns) == 0:
        return 'É necessário discretizar as variáveis numéricas.'

    # Ordenar target se diferente de None
    if order_target:
        df_plot[target] = pd.Categorical(df_plot[target], categories=order_target)
    else:
        df_plot = df_plot

    # Definir subplot
    cols = 1
    rows = math.ceil(len(df_plot.columns) / cols)

    # Criar a figura e os subplots
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(8, 3.25 * rows)) 

    # Flattening os eixos para facilitar o loop
    axes = axes.flatten()

    # Loop para criar um boxplot para cada variável numérica
    for i, col in enumerate(df_plot.columns):
        try:
            if df[target].dtype == object or df[target].dtype == 'category':
                sns.boxplot(data=df_plot, y=df_plot[col], x=df[target], ax=axes[i], orient='v', palette='Blues')
            else:
                sns.boxplot(data=df_plot, x=df_plot[col], y=df[target], ax=axes[i], orient='v', palette='Blues')
        
        except Exception as e:
            print(f"Erro ao plotar {col}: {e}")
            continue
            
        axes[i].set_title(col, fontsize=14)
        axes[i].set(xlabel=None)  

     # Remover os subplots vazios (caso o número de variáveis seja menor que o número de subplots)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Layout
    plt.suptitle('Categorical Features vs Target\n', fontsize=18)
    plt.tight_layout()

    return plt.show()


# Visualizar correlações com a variável target categórica utilizando boxplot
def box_plot_correlations_cat(df, target, order_target): 
    
    # Selecionar colunas para plotagem
    df_plot = df.select_dtypes(include=['number', 'float', 'int'])
    
    if len(df_plot.columns) == 0:
        return 'Não há variáveis numéricas para plotar.'

    # Definir subplot
    cols = 1
    rows = math.ceil(len(df_plot.columns) / cols)

    # Criar a figura e os subplots
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 4 * rows)) 

    # Flattening os eixos para facilitar o loop
    axes = axes.flatten()

    # Criar uma cópia do DataFrame para ordenação
    df_ordered = df.copy()
    if df[target].dtype == object or df[target].dtype == 'category':
        df_ordered[target] = pd.Categorical(df_ordered[target], categories=order_target, ordered=True)
    else:
        df_ordered = df_ordered.sort_values(target)

    # Loop para criar um boxplot para cada variável numérica
    for i, col in enumerate(df_plot.columns):
        try:
            
            sns.boxplot(data=df_ordered, y=col, x=target, ax=axes[i], orient='v', palette='Blues', order=order_target)
            
            axes[i].set_title(col, fontsize=14)
            axes[i].set_xlabel(None)
            axes[i].set_ylabel(None)
            
            # Rotacionar os rótulos do eixo x para melhor legibilidade
            axes[i].tick_params(axis='x', rotation=45)

        except Exception as e:
            print(f"Erro ao plotar {col}: {e}")
            continue

    # Remover os subplots vazios
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Layout
    plt.suptitle(f'Variáveis Numéricas vs {target}\n', fontsize=18)
    plt.tight_layout()

    return plt.show()


# Visualizar correlações com a variável target categórica utilizando boxplot do plotly
def box_plots_corr_cat(df, target, order_target):
    
    # Selecionar colunas numéricas para plotagem
    df_plot = df.select_dtypes(include=['number', 'float', 'int'])
    
    if len(df_plot.columns) == 0:
        print('Não há variáveis numéricas para plotar.')
        return

    # Definir uma paleta de cores formal e adequada para visualização de dados
    color_palette = px.colors.qualitative.Prism

    # Criar subplots
    fig = make_subplots(rows=len(df_plot.columns), cols=1, 
                        subplot_titles=df_plot.columns,
                        vertical_spacing=0.05)  # Reduzir o espaçamento vertical

    # Iterar sobre as colunas numéricas
    for i, col in enumerate(df_plot.columns, 1):
        for j, category in enumerate(order_target):
            trace = go.Box(
                y=df[df[target] == category][col],
                name=category,
                marker_color=color_palette[j % len(color_palette)],
                showlegend=False
            )
            fig.add_trace(trace, row=i, col=1)

        # Atualizar os eixos
        fig.update_xaxes(title_text=target if i == len(df_plot.columns) else None, 
                         ticktext=order_target, tickvals=list(range(len(order_target))), 
                         row=i, col=1)
        fig.update_yaxes(title_text=col, row=i, col=1)

    # Atualizar o layout geral
    fig.update_layout(
        title={
            'text': f'Variáveis Numéricas vs {target}',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=400 * len(df_plot.columns),  # Ajustar a altura total
        width=900,
        #boxmode='group',
        template='plotly_white'
    )

    # Mostrar o gráfico
    fig.show()


# Visualizar correlações com a variável target utilizando boxplot
def box_plot_corr_bucket(df, target): 
    
    # filter df by buckts
    bckt_cols = [col for col in df.columns if 'escala' in col]

    if len(bckt_cols) == 0:
        return "Não existem variáveis categóricas no dataset"

    elif df[target].dtype == object or df[target].dtype == 'category':
        return "A variável target é categórica"

    else:

        df_plot = df[bckt_cols]
        
        # Definir subplot
        cols = 1
        rows = math.ceil(len(df_plot.columns) / cols)
        
        # Criar a figura e os subplots
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(8, 3.25 * rows)) 
        
        # Flattening os eixos para facilitar o loop
        axes = axes.flatten() if len(bckt_cols) != 1 else [axes]
        
        # Loop para criar um boxplot para cada variável numérica
        for i, col in enumerate(df_plot.columns):
            if target in col:
                pass
            else:
                sns.boxplot(data=df_plot, x=df_plot[col], y=df[target], ax=axes[i], orient='v', palette='Blues')
                axes[i].set_title(col, fontsize=14)
                
                # Remover os rótulos do eixo x
                axes[i].set(xlabel=None)  
                
         # Remover os subplots vazios (caso o número de variáveis seja menor que o número de subplots)
        for j in range(i, len(axes)):
            fig.delaxes(axes[j])
       
        # Layout
        plt.suptitle('Categorical Features vs Target\n', fontsize=18)
        plt.tight_layout()
        
        return plt.show()


# Plotar correlações entre variáveis numéricas usando heatmap
def heatmap_correlations(df, type): 
    
    # Filtar variáveis numéricas
    df_plot = df.select_dtypes(exclude=[object, 'category'])
    
    # Calcular correlações
    correlation = df_plot.corr(type)
    
    # Criar a figura e os subplots
    plt.figure(figsize=(15, 8))
    
    # Criar gráfico
    sns.heatmap(correlation, annot=True, vmin=-1, vmax=1, linewidths=0, cmap=palette, alpha=0.75)
    
    # Layout
    plt.title('Numeric Features Correlations\n', fontsize=18)
    
    plt.show()


# Ordenar correlações com a variável target
def heatmap_corr_sorted(df, target, type): 
    
    if df[target].dtype == object or df[target].dtype == 'category':
        return "A variável target é categórica"

    else:
        # Filtar variáveis numéricas
        df_plot = df.select_dtypes(exclude=[object, 'category'])
        
        # Calcular correlações
        correlation = df_plot.corr(type)
        
        # Criar a figura e os subplots
        plt.figure(figsize=(10, 8))
        
        # Criar gráfico
        sns.heatmap(
            correlation[[target]].sort_values(by=target, ascending=False), 
            annot=True, vmin=-1, vmax=1, linewidths=0, cmap=palette, alpha=0.85)
        
        # Layout
        plt.title('Sorted Target Correlations\n', fontsize=18)
        
        plt.show()


# Plotar correlações entre variáveis numéricas usando pairplot
def pairplot_correlations(df): 
    
    sns.set_palette(palette)
    sns.set(style="ticks")
    sns.pairplot(df, diag_kind='kde', height=2, aspect=1.5)
    plt.suptitle('PairPlot Correlations\n', y=1.02, fontsize=20)
    plt.show()


# Select and remove outlier
def remove_outliers(df):
    
    indices = [x for x in df.index]    
    out_indexlist = []
    
    for col in df.columns:
        
        if col in (df.select_dtypes(exclude = [object, 'category']).columns.to_list()):
            Q3 = np.quantile(df[col], 0.75)
            Q1 = np.quantile(df[col], 0.25)
            IQR = Q3 - Q1   
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

            outliers_index = df[col][(df[col] < lower) | (df[col] > upper)].index.tolist()
            outliers = df[col][(df[col] < lower) | (df[col] > upper)].values        
            out_indexlist.extend(outliers_index)

        # Using set to remove duplicates
        out_indexlist = list(set(out_indexlist))
        clean_data = np.setdiff1d(indices, out_indexlist)
        
        df_clean = df[df.index.isin(clean_data)].reset_index(drop=True)

    return df_clean