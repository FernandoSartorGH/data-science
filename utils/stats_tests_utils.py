# Imports
import pandas as pd

# Statistic
import pingouin as pg
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro, kstest, probplot, anderson, normaltest
from statsmodels.stats.diagnostic import lilliefors, het_goldfeldquandt

from scipy.stats import shapiro, jarque_bera
from statsmodels.tsa.stattools import adfuller

from statsmodels.stats.stattools import durbin_watson, omni_normtest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, acorr_ljungbox, linear_reset


# Normal test
def normal_test(alpha, df):
    alpha = alpha
    names = []
    test = []
    p_value = []
    
    print(f"Alpha: {alpha}")
    print()
    print("Hipotese nula: Os dados seguem distribuição normal")
    print("Hipotese alternativa: Os dados não seguem distribuição normal")
        
    
    for col in df.select_dtypes(exclude=[object, 'category']).columns:
        k2, p = normaltest(df[col])
        names.append(col)
        test.append('Rejeita H0. A distribuição dos dados NÃO é normal.' if p < alpha else 'Não rejeita H0. A distribuição é normal.')
        p_value.append(p.round(6))
        
    return pd.DataFrame({'Name': names,
                         'Test': test,
                         'p-value': p_value})


# Normal test
def residual_normal_test(errors):

    print("H0: Os dados seguem distribuição normal")
    print("H1: Os dados não seguem distribuição normal")
    print('-'*55)
    print()

    # Teste de normalidade Shapiro-Wilk
    stat_shapiro, p_value_shapiro = shapiro(errors)
    print('Teste de Shapiro-Wilk')
    print(f" estatística de teste: {round(stat_shapiro, 4)} | p_value: {round(p_value_shapiro, 4)}")
    print(' Rejeita H0, os dados não seguem distribuição normal') if p_value_shapiro < 0.05 else print(' Não rejeita H0, os dados seguem distribuição normal')
    print('-'*55)

    # Teste de normalidade Kolmogorov-Smirnov
    stat_ks, p_value_ks = kstest(errors, 'norm')
    print('Teste de Kolmogorov-Smirnov')
    print(f" estatística de teste: {round(stat_ks, 4)} | p_value: {round(p_value_ks, 4)}")
    print(' Rejeita H0, os dados não seguem distribuição normal') if p_value_ks < 0.05 else print(' Não rejeita H0, os dados seguem distribuição normal')
    print('-'*55)

    # Teste de normalidade Liliefors
    stat_ll, p_value_ll = lilliefors(errors, 'norm')
    print('Teste de Liliefors')
    print(f" estatística de teste: {round(stat_ll, 4)} | p_value: {round(p_value_ll, 4)}")
    print(' Rejeita H0, os dados não seguem distribuição normal') if p_value_ll < 0.05 else print(' Não rejeita H0, os dados seguem distribuição normal')
    print('-'*55)

    # Teste de normalidade Anderson
    stat_and, critical_and, significance_and = anderson(errors.values.flatten(), dist='norm')
    print('Teste de Anderson')
    print(f" estatística de teste: {round(stat_and, 4)} | Valor crítico: {round(critical_and[2], 4)}")
    print(' Rejeita H0, os dados não seguem distribuição normal') if critical_and[2]< stat_and else print(' Não rejeita H0, os dados seguem distribuição normal')
    print('-'*55)


# Teste chi2 para independência entre as variáveis
def independence_chi2_test(df, target):
    """
    
    Features independence (chi2)
      H0: As variáveis são independentes
      H1: As variáveis não são independentes
      se p > 0.05 Aceita H0

    valor_esperado: Frequência que seria esperada se não houvesse associação entre as variáveis calculada assumindo a distribuição do teste
    valor_observado: Frequência real dos dados do df
    
    """

    # Empty lists
    p_value = []
    cols = []
    results = []

    # print Hipoteses
    print('H0: As variáveis são independentes')
    print('H1: As variáveis não são independentes')
    print('Alpha: 0.05')

    # Iter
    for col in df.columns:
        valor_esperado, valor_observado, estatisticas = pg.chi2_independence(df, target, col)

        pval = estatisticas[estatisticas['test']=='pearson']['pval'][0]
        if pval > 0.05:
            result = 'Aceita H0, as variáveis são independentes'
        else:
            result = 'Rejeita H0, as variáveis não são independentes'

        p_value.append(pval)
        cols.append(col)
        results.append(result)

    # Data Frame
    data_df = {
        'feature': cols,
        'pval': p_value,
        'result': results
    }

    df_result = pd.DataFrame(data_df)

    return df_result