#Imports
import seaborn as sns
from numpy import polyfit
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# Plot dist 2x2
def residual_plot(y_pred, errors, title):

    # Plot
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 8))
    fig.suptitle(f"{title}", fontsize=18)

    # Distribuição
    ax[0,0].hist(errors)
    ax[0,0].set_title("Histograma")
    ax[0,1].boxplot(errors)
    ax[0,1].set_title("Boxplot")

    # Scatter e linha de tendência
    ax[1,0].scatter(y_pred, errors)
    ax[1,0].set_title("Homoscedasticity")

    # Auto correlação
    plot_acf(errors, ax = ax[1,1])

    plt.ticklabel_format(style='plain', axis='y')

    # Set tiltle
    #plt.title(f"{title}", fontsize=14)

    # Remove x, y Ticks
    plt.grid(color = 'silver', linestyle = '-', linewidth = 0.5, axis = 'y', alpha = 0.2)

    # Remove x, y Ticks
    for i in [0,1]:
      for j in [0,1]:
        ax[i,j].xaxis.set_ticks_position('none')
        ax[i,j].yaxis.set_ticks_position('none')

        # Remove axes splines
        for s in ['top', 'left', 'right']:
            ax[i,j].spines[s].set_visible(False)

    # Adiciona linhas horizontais ao subplot de resíduos
    ax[1,0].axhline(y=0, color='darkslategray', linestyle='--', alpha=0.5)
    ax[1,0].axhline(y=2, color='slategray', linestyle='--', alpha=0.5)
    ax[1,0].axhline(y=-2, color='slategray', linestyle='--', alpha=0.5)

    # Plot
    plt.subplots_adjust(hspace=0.5)
    plt.show()
