import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import fminbound


# ---------- cálculo de ciclo e eficiências ----------

def calculate_efficiencies(df):
    # ENERGIAS INTERNAS
    df['U1B'] = df['H1B'] - df['P1B'] / df['RHO1B']
    df['U2B'] = df['H2B'] - df['P2B'] / df['RHO2B']
    df['U3B'] = df['H3B'] - df['P3B'] / df['RHO3B']
    df['U4B'] = df['H4B'] - df['P4B'] / df['RHO4B']

    # CALORES TOTAIS
    df['QCOMP_total'] = df['T1B'] * (df['S1B'] - df['S4B']) * df['MCICLO']
    df['QAQUEC_total'] = df['MCICLO'] * (df['U2B'] - df['U1B'])
    df['QTURB_total'] = df['T3B'] * (df['S3B'] - df['S2B']) * df['MCICLO']
    df['QRESF_total'] = df['MCICLO'] * (df['U4B'] - df['U3B'])

    # VARIAÇÕES
    df['deltaU_COMP'] = (df['U1B'] - df['U4B']) * df['MCICLO']
    df['deltaU_TURB'] = (df['U3B'] - df['U2B']) * df['MCICLO']
    df['WCOMP'] = df['QCOMP_total'] - df['deltaU_COMP']
    df['WTURB'] = df['QTURB_total'] - df['deltaU_TURB']
    df['Wnet'] = abs(df['WTURB'] - df['WCOMP'])

    # Cp MÉDIOS
    df['Cp1B'] = (df['H2B'] - df['H1B']) / (df['T2B'] - df['T1B'])
    df['Cp3B'] = (df['H4B'] - df['H3B']) / (df['T4B'] - df['T3B'])

    # REGENERADOR E OTIMIZAÇÃO
    def func_pinch(DeltaT_pinch, dataframe):
        T1B_int = (
            dataframe['Cp3B'] * (dataframe['T3B'] - DeltaT_pinch) +
            dataframe['Cp1B'] * dataframe['T1B']
        ) / (dataframe['Cp1B'] + dataframe['Cp3B'])

        Qregen = dataframe['MCICLO'] * dataframe['Cp1B'] * (T1B_int - dataframe['T1B'])
        Qregen = Qregen.clip(lower=0)

        Qreal = dataframe['QTURB_total'] + dataframe['QAQUEC_total'] - Qregen
        Ef_regen = (dataframe['Wnet'] / Qreal) * 100
        Ef_regen = Ef_regen.where((Ef_regen >= 0) & (Ef_regen <= 100), np.nan)
        return -Ef_regen.mean()

    DeltaT_otimo = fminbound(func_pinch, 0.1, 30, args=(df,))
    Ef_otimo = -func_pinch(DeltaT_otimo, df)

    print(f"DeltaT ótimo encontrado: {DeltaT_otimo:.4f} K (eficiência {Ef_otimo:.2f}%)")

    # aplicar e calcular eficiências
    df['T1B_int'] = (
        df['Cp3B'] * (df['T3B'] - DeltaT_otimo) + df['Cp1B'] * df['T1B']
    ) / (df['Cp1B'] + df['Cp3B'])
    df['T3B_int'] = df['T1B_int'] + DeltaT_otimo
    df['Qregen'] = (df['MCICLO'] * df['Cp1B'] * (df['T1B_int'] - df['T1B'])).clip(lower=0)
    df['Qreal'] = df['QAQUEC_total'] - df['Qregen']

    df['Efcarnot'] = (1 - df['T1B'] / df['T3B']) * 100
    df['Ef_regen'] = (df['Wnet'] / (df['QTURB_total'] + df['Qreal'])) * 100
    df['Ef_sregen'] = (df['Wnet'] / (df['QTURB_total'] + df['QAQUEC_total'])) * 100

    mask = (
        (df['Efcarnot'] >= 0) & (df['Efcarnot'] <= 100) &
        (df['Ef_regen'] >= 0) & (df['Ef_regen'] <= 100) &
        (df['Ef_sregen'] >= 0) & (df['Ef_sregen'] <= 100) &
        (df['Ef_regen'] <= df['Efcarnot']) &
        (df['Ef_sregen'] <= df['Efcarnot'])
    )
    
    # DEBUG: Ver quantas linhas passaram no filtro
    print(f"\n=== DEBUG ===")
    print(f"Total de linhas originais: {len(df)}")
    print(f"Linhas que passaram no filtro: {mask.sum()}")
    print(f"\nAnálise de cada critério:")
    print(f"  Efcarnot válida (0-100): {((df['Efcarnot'] >= 0) & (df['Efcarnot'] <= 100)).sum()}")
    print(f"  Ef_regen válida (0-100): {((df['Ef_regen'] >= 0) & (df['Ef_regen'] <= 100)).sum()}")
    print(f"  Ef_sregen válida (0-100): {((df['Ef_sregen'] >= 0) & (df['Ef_sregen'] <= 100)).sum()}")
    print(f"  Ef_regen <= Efcarnot: {(df['Ef_regen'] <= df['Efcarnot']).sum()}")
    print(f"  Ef_sregen <= Efcarnot: {(df['Ef_sregen'] <= df['Efcarnot']).sum()}")
    print(f"\nValores das eficiências:")
    print(f"  Efcarnot: min={df['Efcarnot'].min():.2f}, max={df['Efcarnot'].max():.2f}")
    print(f"  Ef_regen: min={df['Ef_regen'].min():.2f}, max={df['Ef_regen'].max():.2f}")
    print(f"  Ef_sregen: min={df['Ef_sregen'].min():.2f}, max={df['Ef_sregen'].max():.2f}")
    print(f"=============\n")
    
    df_filtrado = df[mask].copy()
    
    if len(df_filtrado) == 0:
        print("⚠️  AVISO: Nenhuma linha passou no filtro! Exportando dados SEM filtro...")
        df_filtrado = df.copy()
    
    return df_filtrado, DeltaT_otimo, Ef_otimo


# ---------- funções de plotagem para mapas ----------

def _pivot_efficiency(df, col_x, col_y, eff_col):
    x_vals = np.sort(df[col_x].unique())
    y_vals = np.sort(df[col_y].unique())
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.full((len(y_vals), len(x_vals)), np.nan)
    for _, row in df.iterrows():
        ix = np.where(x_vals == row[col_x])[0]
        iy = np.where(y_vals == row[col_y])[0]
        if ix.size and iy.size:
            Z[iy[0], ix[0]] = row[eff_col]
    Z[Z <= 0] = np.nan
    return x_vals, y_vals, Z


def plot_pressure_map(df, eff_col='Ef_regen', output='Mapa_Eficiencia_Pressao.pdf'):
    Pcomp_vals, Pturb_vals, Ef_matrix = _pivot_efficiency(df, 'Pcomp', 'Pturb', eff_col)
    Ef_min = np.nanmin(Ef_matrix)
    Ef_max = np.nanmax(Ef_matrix)

    fig, ax = plt.subplots(figsize=(20/2.54, 14/2.54), dpi=150)
    contourf = ax.contourf(Pcomp_vals, Pturb_vals, Ef_matrix, levels=np.linspace(Ef_min, Ef_max, 50),
                           cmap='turbo', extend='both', antialiased=True)
    contour = ax.contour(Pcomp_vals, Pturb_vals, Ef_matrix,
                         levels=np.linspace(Ef_min, Ef_max, 12),
                         colors='k', linewidths=0.8)
    ax.clabel(contour, inline=True, fontsize=10, fmt='%.2f')
    cb = fig.colorbar(contourf, ax=ax, ticks=np.linspace(Ef_min, Ef_max, 6))
    from matplotlib import ticker
    cb.ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    cb.set_label('Efficiency (%)',fontsize=14)
    ax.set_xlabel('Compressure pressure', fontsize=14)
    ax.set_ylabel('Turbine pressure', fontsize=14)
    ax.set_title('Efficiency map', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.patch.set_facecolor('white')
    ax.tick_params(labelsize=14, direction='in')
    ax.set_aspect('auto')

    # Tornar as 4 bordas (spines) mais espessas e pretas
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_edgecolor('black')
    # Garantir que o contorno da colorbar seja visível e espesso
    try:
        cb.outline.set_linewidth(2.0)
        cb.outline.set_edgecolor('black')
    except Exception:
        pass

    # Adicionar uma borda interna (retângulo) para facilitar a visualização
    from matplotlib.patches import Rectangle
    rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor='none',
                     edgecolor='black', linewidth=2.0, zorder=10)
    ax.add_patch(rect)

    # Exibir gráfico interativo para ajustes
    plt.show()
    
    # Após fechar a janela interativa, salvar como PDF
    response = input(f"\nDeseja salvar o gráfico como '{output}'? (s/n): ").strip().lower()
    if response == 's':
        fig.savefig(output, format='pdf', bbox_inches='tight')
        print(f"Gráfico salvo como '{output}'")
    else:
        print(f"Gráfico '{output}' não foi salvo.")
    plt.close(fig)


def plot_temperature_map(df, eff_col='Ef_regen', output='Mapa_Eficiencia_Temperatura.pdf'):
    # x-axis = heater temperature (Tresf), y-axis = cooler temperature (Taqc)
    Tresf_vals, Taqc_vals, Ef_matrix = _pivot_efficiency(df, 'Tresf', 'Taqc', eff_col)
    Ef_min = np.nanmin(Ef_matrix)
    Ef_max = np.nanmax(Ef_matrix)

    fig, ax = plt.subplots(figsize=(20/2.54, 14/2.54), dpi=150)
    # use returned x and y in correct order
    contourf = ax.contourf(Tresf_vals, Taqc_vals, Ef_matrix, levels=np.linspace(Ef_min, Ef_max, 50),
                           cmap='turbo', extend='both', antialiased=True)
    contour = ax.contour(Tresf_vals, Taqc_vals, Ef_matrix,
                         levels=np.linspace(Ef_min, Ef_max, 12),
                         colors='k', linewidths=0.8)
    ax.clabel(contour, inline=True, fontsize=10, fmt='%.2f')
    cb = fig.colorbar(contourf, ax=ax, ticks=np.linspace(Ef_min, Ef_max, 6))
    from matplotlib import ticker
    cb.ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    cb.set_label('Efficiency (%)', fontsize=14)
    ax.set_xlabel('Temperature cooler', fontsize=14)
    ax.set_ylabel('Temperature heater', fontsize=14)
    ax.set_title('Efficiency map',fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.patch.set_facecolor('white')
    ax.tick_params(labelsize=14, direction='in')
    ax.set_aspect('auto')

    # Tornar as 4 bordas (spines) mais espessas e pretas
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_edgecolor('black')
    # Garantir que o contorno da colorbar seja visível e espesso
    try:
        cb.outline.set_linewidth(2.0)
        cb.outline.set_edgecolor('black')
    except Exception:
        pass

    # Adicionar uma borda interna (retângulo) para facilitar a visualização
    from matplotlib.patches import Rectangle
    rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor='none',
                     edgecolor='black', linewidth=2.0, zorder=10)
    ax.add_patch(rect)

    # Exibir gráfico interativo para ajustes
    plt.show()
    
    # Após fechar a janela interativa, salvar como PDF
    response = input(f"\nDeseja salvar o gráfico como '{output}'? (s/n): ").strip().lower()
    if response == 's':
        fig.savefig(output, format='pdf', bbox_inches='tight')
        print(f"Gráfico salvo como '{output}'")
    else:
        print(f"Gráfico '{output}' não foi salvo.")
    plt.close(fig)


# ---------- fluxo principal ----------

if __name__ == '__main__':
    df = pd.read_excel('dados.xlsx')
    df_filtrado, deltaT, ef_opt = calculate_efficiencies(df)

    # exporta resultados em um único arquivo Excel
    arquivo_saida = 'Resultado_Eficiencias.xlsx'
    with pd.ExcelWriter(arquivo_saida) as writer:
        df_filtrado.to_excel(writer, sheet_name='Geral', index=False)
        for Tfix in df_filtrado['T4B'].unique():
            df_filtrado[df_filtrado['T4B'] == Tfix].to_excel(
                writer, sheet_name=f'Tresf_{int(Tfix)}K', index=False)
    print(f'Arquivo {arquivo_saida} criado com todas as eficiências (inclui Ef_regen e Ef_sregen).')

    # gerar mapas para ambos os casos
    plot_pressure_map(df_filtrado, eff_col='Ef_regen', output='Mapa_Eficiencia_Pressao_ComReg.pdf')
    plot_pressure_map(df_filtrado, eff_col='Ef_sregen', output='Mapa_Eficiencia_Pressao_SemReg.pdf')
    plot_temperature_map(df_filtrado, eff_col='Ef_regen', output='Mapa_Eficiencia_Temperatura_ComReg.pdf')
    plot_temperature_map(df_filtrado, eff_col='Ef_sregen', output='Mapa_Eficiencia_Temperatura_SemReg.pdf')