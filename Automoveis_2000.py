# Importando bibliotecas necessárias
import pandas as pd
from pycaret.time_series import TSForecastingExperiment
import streamlit as st

# Configuração da página do Streamlit
st.set_page_config(
    page_title='Forecast Licenciamentos Automóveis',
    layout='wide',
    initial_sidebar_state='auto'
)

# Cabeçalho da página
st.header("Forecast Licenciamentos de Automóveis", divider='violet')

# Layout principal
col1, col2, col3 = st.columns([1, 1, 1], gap='large')

with col1:
    st.subheader('Forecast Licenciamentos Automóveis por meio do PyCaret', divider='violet')
    with st.expander('Explicação mercado de Automóveis'):
        st.markdown('''## O Mercado Automobilístico Brasileiro Segundo a Anfavea

A Associação Nacional dos Fabricantes de Veículos Automotores (Anfavea) é a principal entidade que reúne as empresas fabricantes de automóveis no Brasil. Seus dados e análises são cruciais para entender a dinâmica do setor automotivo no país.

**Tendências Recentes:**

Nos últimos anos, o mercado automobilístico brasileiro tem apresentado um cenário bastante dinâmico, influenciado por diversos fatores como:

* **Crise econômica:** Períodos de crise econômica impactaram negativamente as vendas, com os consumidores adiando a compra de veículos.
* **Juros altos:** As altas taxas de juros dificultam o acesso ao crédito, afetando a demanda por veículos financiados.
* **Mudanças nas preferências do consumidor:** A busca por veículos mais eficientes e com tecnologias mais avançadas tem moldado a oferta das montadoras.
* **Política industrial:** As políticas governamentais, como incentivos fiscais e regulamentações ambientais, influenciam diretamente a produção e as vendas de veículos.

**Dados da Anfavea:**

A Anfavea publica regularmente dados sobre produção, vendas, exportações e importações de veículos. Esses dados permitem acompanhar a evolução do setor e identificar tendências.

Algumas das principais informações divulgadas pela Anfavea incluem:

* **Produção:** Dados sobre a quantidade de veículos produzidos no Brasil, por tipo (carros, comerciais leves, caminhões, ônibus) e por fabricante.
* **Vendas:** Informações sobre as vendas de veículos no mercado interno, tanto para consumidores finais quanto para frotas.
* **Exportações e importações:** Dados sobre o volume de veículos exportados e importados, bem como os principais destinos e origens.
* **Emprego:** Informações sobre o número de empregos gerados pelo setor.

**Desafios e Oportunidades:**

O mercado automobilístico brasileiro enfrenta diversos desafios, como a concorrência cada vez mais acirrada, a necessidade de investir em novas tecnologias e a pressão por maior sustentabilidade.

Por outro lado, existem diversas oportunidades, como:

* **Crescimento da classe média:** O aumento da renda da classe média pode impulsionar a demanda por veículos.
* **Urbanização:** A crescente urbanização exige soluções de mobilidade mais eficientes, o que pode impulsionar a demanda por veículos elétricos e compartilhados.
* **Novas tecnologias:** O desenvolvimento de tecnologias como a condução autônoma e a conectividade pode abrir novas oportunidades de negócios.

**Para obter informações mais detalhadas e atualizadas sobre o mercado automobilístico brasileiro, recomendo acessar o site da Anfavea:** [https://www.anfavea.com.br/](https://www.anfavea.com.br/)

**Gostaria de saber mais sobre algum aspecto específico do mercado automobilístico brasileiro?** Por exemplo, posso fornecer informações sobre:

* **O impacto da pandemia de COVID-19 no setor.**
* **As perspectivas para os próximos anos.**
* **As principais montadoras que atuam no Brasil.**
* **Os programas de incentivo à produção de veículos elétricos.**

**Se tiver alguma outra pergunta, fique à vontade para perguntar!**
''')

with col2:
    st.subheader("**Utilizando PyCaret**", divider='violet')
    with st.expander('Explicação PyCaret'):
        st.markdown('''**O módulo PyCaret Time Series é uma ferramenta avançada para analisar e prever dados de séries temporais usando aprendizado de máquina e técnicas estatísticas clássicas. Esse módulo permite que os usuários executem facilmente tarefas complexas de previsão de séries temporais, automatizando todo o processo, desde a preparação dos dados até a implantação do modelo.

O PyCaret Time Series Forecasting oferece suporte a métodos como ARIMA, Prophet e LSTM, além de ferramentas para lidar com valores ausentes, decomposição de séries temporais e visualizações de dados**.
''')
    
with col3:
    st.subheader('Imagem Automóveis (Anfavea)', divider='violet')
    st.image(
        'https://th.bing.com/th/id/OIP.Y0-HA7TCU9TvyotGAodPxwHaES?rs=1&pid=ImgDetMain',
        use_container_width=True
    )

# Carregando o arquivo Excel local
try:
    data = pd.read_excel(r"C:\Tablets\Automoveis_2000.xlsx")
    data['Mês'] = pd.to_datetime(data['Mês'])  # Ajustar o nome da coluna de data, se necessário
    data.set_index('Mês', inplace=True)   # Definir a coluna de data como índice
    st.success("Base de dados carregada com sucesso.")         
except Exception as e:
    st.error(f"Erro ao carregar a base de dados: {e}")
    st.stop()    
    

# Visualizar a base de dados no Streamlit
col1, col2 = st.columns([1, 1], gap='large')

with col1:
    st.write('***Base de Dados Anfavea***')
    st.dataframe(data, use_container_width=True)    

with col2:
    # Configuração inicial do experimento
    s = TSForecastingExperiment()
    s.setup(data=data, target='AUTOMÓVEIS', session_id=123)
    st.write("**Configuração inicial do PyCaret concluída.**")

    # Comparar modelos
    best = s.compare_models()


    # Obter a tabela de comparação
    comparison_df_sl = s.pull()
    st.write("### Comparação de Modelos")
    st.dataframe(comparison_df_sl)

    # Botão para download da tabela de comparação
    csv_me = comparison_df_sl.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Baixar Comparação",
        data=csv_me,
        file_name="model_comparison.csv",
        mime="text/csv",
        key="download_button_comparison_sl"
    )

# Seção para visualização e previsões
col1, col2, col3 = st.columns([1, 1, 1], gap='large')

with col1:
    st.write('**Time Series - Target = Automóveis**')
    s.plot_model(best, plot='ts', display_format='streamlit')  

with col2:
    # Finalizar o modelo
    final_best = s.finalize_model(best)    
    st.write("**Modelo finalizado:**")
    st.write(final_best)

with col3:
    # Plotar previsões
    st.write("**Previsão com horizonte de 36 períodos:**")
    s.plot_model(final_best, plot='forecast', data_kwargs={'fh': 36}, display_format='streamlit')

# Exibindo previsões e métricas
col1, col2, col3 = st.columns([3, 1, 1], gap='large')

with col1:
    predictions = s.predict_model(final_best, fh=36)
    st.write("**Previsões:**")
    st.dataframe(predictions, use_container_width=True)
    
    # Botão para download das previsões
    csv = predictions.to_csv(index=False).encode('utf-8')
    st.download_button("Baixar Previsão",
                        data=csv, 
                        file_name="predictions.csv",
                        mime='text/csv',
                        key='download_button_previsao_sl')

with col2:
    st.write('Previsão 2025')
    st.metric(label='2025: Previsão e Variação', value=2688487, delta=7.4)

with col3:
    st.write('Previsão 2026')
    st.metric(label='2026: Previsão e Variação', value=2738146, delta=1.8)


col1, col2=st.columns([1,1], gap='large')

with col1:
    with st.expander('Explicação CatBoost Regressor w/ Cond. Deseasonalize & Detrending'):
        st.markdown('''O modelo CatBoost Regressor w/ Conditional Deseasonalization & Detrending aplicado à previsão de séries temporais envolve um processo de preparação dos dados para melhorar a capacidade de previsão, especificamente focando em remover componentes sazonais e tendências antes de aplicar o modelo de aprendizado de máquina (CatBoost). Vamos decompor o modelo e entender o que cada parte significa.
1. CatBoost Regressor
O CatBoost é um modelo de aprendizado de máquina baseado em árvores de decisão, desenvolvido pela Yandex. Ele é especialmente eficaz em lidar com variáveis categóricas e pode ser utilizado para tarefas de regressão (previsão contínua) e classificação. Ele tem a vantagem de ser eficiente no tratamento de dados desbalanceados, categóricos e com características complexas, tornando-o popular para previsões em séries temporais.
Em uma tarefa de regressão, o CatBoost aprende a mapear uma série de entradas (features) para uma variável de saída contínua. Para séries temporais, as variáveis de entrada podem incluir dados históricos, variáveis externas (exógenas), e atributos como data e hora.
2. Deseasonalization (Desseasonalização)
A deseasonalização é o processo de remover o componente sazonal de uma série temporal. As séries temporais frequentemente exibem padrões sazonais que ocorrem em intervalos regulares (por exemplo, vendas de produtos que aumentam durante as festas de fim de ano). Quando esses padrões sazonais não são removidos, o modelo pode ter dificuldades para aprender as tendências subjacentes e fazer boas previsões.
A deseasonalização é feita dividindo a série temporal pela sua média sazonal ou aplicando um filtro para remover a variação sazonal, o que deixa apenas a tendência e o "ruído". Após o modelo fazer previsões, a sazonalidade pode ser restaurada, recompondo os dados de previsão.
3. Detrending (Remoção de Tendência)
A remoção de tendência (detrending) é outro passo importante no pré-processamento de séries temporais. Muitas séries temporais têm uma tendência de longo prazo, ou seja, uma direção consistente (crescente ou decrescente) nos dados ao longo do tempo. A tendência pode distorcer a previsão, pois o modelo pode acabar tentando aprender a tendência ao invés de se concentrar nas flutuações de curto prazo.
A remoção da tendência é feita através de métodos como:
•	Diferença: Subtrair o valor de uma observação com a do período anterior.
•	Métodos de suavização: Como médias móveis, para capturar apenas as flutuações ao redor da tendência.
Após a remoção da tendência, o modelo se concentrará em aprender as variações sazonais e cíclicas mais finas e temporais.
4. Combinando CatBoost, Deseasonalization e Detrending
Quando usamos CatBoost para séries temporais, ele pode aprender padrões muito complexos, mas, se as séries apresentarem forte sazonalidade ou tendência, o modelo pode não funcionar bem em sua forma bruta. A combinação do CatBoost com os processos de deseasonalização e detrending pode melhorar a performance do modelo. Aqui está como isso funciona:
1.	Deseasonalização: Antes de alimentar os dados no CatBoost, você pode remover os padrões sazonais para que o modelo se concentre nas tendências mais gerais e nos dados residuals (sem a variação sazonal).
2.	Detrending: Após a deseasonalização, o próximo passo é remover a tendência de longo prazo. Isso deixa o modelo para aprender apenas as flutuações mais finas dos dados.
3.	Aplicação do CatBoost: Com os dados já tratados (sem sazonalidade e sem tendência), o CatBoost pode aprender de forma mais eficaz os padrões de curto prazo e prever os valores futuros com maior precisão.
4.	Restaurando Sazonalidade e Tendência: Após o CatBoost gerar as previsões para a série temporal "detrended" e "deseasonalized", você pode restaurar a sazonalidade e a tendência removidas anteriormente, para obter previsões mais realistas e de longo prazo.
Vantagens:
•	Aumento da precisão: O CatBoost pode aprender padrões mais sutis e complexos quando os dados são preparados sem as variações sazonais e tendências de longo prazo.
•	Redução do erro de previsão: Ao remover sazonalidade e tendência, o modelo pode capturar mais precisamente as flutuações de curto prazo, resultando em previsões melhores.
Este processo é particularmente útil em casos de séries temporais com fortes tendências ou sazonalidades, pois permite que o modelo se concentre nos componentes mais relevantes para a previsão.
''', unsafe_allow_html=True)
   
with col2:
    with st.expander('**Análise dos Resultados**'):
        st.markdown('''Os resultados fornecidos indicam a performance do modelo **CatBoost Regressor com Condicional Deseasonalização e Detrend** na tarefa de previsão de séries temporais. Vamos analisar cada uma das métricas para entender o desempenho do modelo.

### 1. **MASE (Mean Absolute Scaled Error)**
   - **Valor**: 0.3158
   - **Interpretação**: O MASE é uma métrica que compara o erro absoluto médio do modelo com o erro absoluto médio de um modelo de referência simples (normalmente uma previsão baseada no histórico). Um MASE abaixo de 1 indica que o modelo tem melhor desempenho do que o modelo de referência.
   - **Análise**: O valor de **0.3158** é bem baixo, sugerindo que o modelo **CatBoost com deseasonalização e detrend** está performando significativamente melhor do que o modelo de referência. Ou seja, o modelo é eficaz na previsão da série temporal.

### 2. **RMSSE (Root Mean Squared Scaled Error)**
   - **Valor**: 0.2374
   - **Interpretação**: O RMSSE é semelhante ao MASE, mas calcula a raiz quadrada do erro quadrático médio. Essa métrica também compara o erro do modelo com um modelo simples. Assim como o MASE, valores abaixo de 1 indicam bom desempenho.
   - **Análise**: O valor de **0.2374** é também muito bom e sugere que o modelo está fornecendo previsões com um erro consideravelmente menor do que o modelo de referência.

### 3. **MAE (Mean Absolute Error)**
   - **Valor**: 9659.9057
   - **Interpretação**: O MAE calcula o erro absoluto médio entre as previsões e os valores reais. Menores valores de MAE indicam melhores previsões, pois o erro médio entre as previsões e os valores reais é menor.
   - **Análise**: O valor de **9659.9057** é uma medida de erro absoluto, e a análise desse número depende da escala dos dados. Se os valores da série temporal forem elevados, o MAE pode ser relativamente grande, mas considerando que esse é o valor médio de erro absoluto, ele deve ser comparado com os valores reais da série para uma interpretação mais precisa.

### 4. **RMSE (Root Mean Squared Error)**
   - **Valor**: 9659.9057
   - **Interpretação**: O RMSE é semelhante ao MAE, mas dá mais peso aos erros maiores devido à sua base quadrada. Ele é sensível a grandes desvios, e valores menores indicam melhor desempenho.
   - **Análise**: O valor de **9659.9057** para o RMSE sugere um erro médio relativamente grande (semelhante ao MAE), mas novamente, esse número precisa ser analisado em relação ao valor da série temporal. Se os valores da série são altos, esse erro pode ser aceitável.

### 5. **MAPE (Mean Absolute Percentage Error)**
   - **Valor**: 0.0394 (ou 3.94%)
   - **Interpretação**: O MAPE calcula o erro percentual absoluto médio, ou seja, a diferença média percentual entre as previsões e os valores reais. Ele é útil quando se quer entender o erro em termos relativos. Valores menores indicam melhor desempenho.
   - **Análise**: O valor de **0.0394** (ou 3.94%) sugere que o modelo está fazendo previsões com um erro percentual médio baixo, o que é uma excelente indicação de boa acurácia. O MAPE abaixo de 5% é normalmente considerado muito bom.

### 6. **SMAPE (Symmetric Mean Absolute Percentage Error)**
   - **Valor**: 0.0411 (ou 4.11%)
   - **Interpretação**: O SMAPE é uma variação do MAPE que trata simetricamente os erros positivos e negativos, ou seja, é uma versão mais balanceada do MAPE. Ele é usado para medir a precisão de previsões de séries temporais. Como o MAPE, valores menores são desejáveis.
   - **Análise**: O valor de **0.0411** (ou 4.11%) também é bom, indicando que o modelo tem um erro percentual simétrico muito baixo, semelhante ao MAPE. Isso reforça a alta precisão do modelo.

### 7. **TT (Tempo Total em Segundos)**
   - **Valor**: 1.5667 segundos
   - **Interpretação**: Esse valor indica o tempo total que o modelo levou para treinar e fazer previsões. Em tarefas de previsão de séries temporais, é importante que o tempo de treinamento e previsão seja razoável, principalmente em ambientes de produção.
   - **Análise**: O tempo de **1.5667 segundos** é muito baixo, indicando que o modelo foi capaz de treinar e fazer previsões rapidamente, o que é um ponto positivo em termos de eficiência.

---

### Conclusão da Análise:

O modelo **CatBoost Regressor com Condicional Deseasonalização e Detrend** apresenta **excelentes resultados** nas métricas de erro:

- **MASE e RMSSE** muito baixos indicam que o modelo está superando o modelo de referência de forma consistente.
- **MAPE e SMAPE baixos** indicam boa precisão relativa, com erros percentuais muito pequenos.
- O **tempo de execução** de apenas **1.5667 segundos** sugere que o modelo é eficiente e adequado para cenários de previsão em tempo real.

Em resumo, o modelo está bem ajustado, com **boa precisão e baixo erro**, sendo uma ótima escolha para tarefas de previsão de séries temporais. Isso indica que o pré-processamento (deseasonalização e detrend) aliado ao uso do CatBoost proporcionou um desempenho excepcional.
''')

#Modelagem Atualizada dezembro/2024

# Cabeçalho da página
st.header("Forecast Licenciamentos de Automóveis - atualizado dezembro 2024", divider='green')

# Layout principal
col1, col2, col3 = st.columns([1, 1, 1], gap='large')

with col1:
    st.subheader('Fenabrave projeta crescimento tímido nas vendas de carros novos em 2025', divider='green')
    with st.expander('A Fenabrave, entidade que representa as concessionárias de veículos, projeta um crescimento de apenas 5% nas vendas para este ano'):
        st.markdown('''Ano de 2024 x projeções para 2025
Segundo a Fenabrave, o ano de 2024 foi marcado pela recuperação do mercado automotivo, com 2,63 milhões de veículos emplacados. A oferta de crédito, a diversificação de produtos e a manutenção dos estoques foram fatores que contribuíram para esse resultado positivo.

Para 2025, além do crescimento de 5% nas vendas de automóveis, a Fenabrave projeta um aumento de 10% nas vendas de motocicletas e de 5% no volume de vendas de implementos rodoviários.
''')
    
with col3:
    st.subheader('Imagem Publicação Fenabrave', divider='violet')
    st.image(r'C:\Tablets\fenabrave.png', use_container_width=True)

try:
    data = pd.read_excel(r"C:\Tablets\Automoveis_2000_2024.xlsx")
    data['Mês'] = pd.to_datetime(data['Mês'])  # Ajustar o nome da coluna de data, se necessário
    data.set_index('Mês', inplace=True)   # Definir a coluna de data como índice
    st.success("Base de dados carregada com sucesso.")         
except Exception as e:
    st.error(f"Erro ao carregar a base de dados: {e}")
    st.stop() 
    
# Visualizar a base de dados no Streamlit
col1, col2 = st.columns([1, 1], gap='large')

with col1:
    st.write('***Base de Dados Anfavea***')
    st.dataframe(data, use_container_width=True)    

with col2:
    # Configuração inicial do experimento
    s = TSForecastingExperiment()
    s.setup(data=data, target='AUTOMÓVEIS', session_id=123)
    st.write("**Configuração inicial do PyCaret concluída.**")

    # Comparar modelos
    best = s.compare_models()

# Obter a tabela de comparação
    comparison_df_sl = s.pull()
    st.write("### Comparação de Modelos")
    st.dataframe(comparison_df_sl)

    # Botão para download da tabela de comparação
    csv_auto = comparison_df_sl.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Baixar Comparação",
        data=csv_auto,
        file_name="model_comparison.csv",
        mime="text/csv",
        key="download_button_comparison_auto"
    )

# Seção para visualização e previsões
col1, col2, col3 = st.columns([1, 1, 1], gap='large')

with col1:
    st.write('**Time Series - Target = Automóveis**')
    s.plot_model(best, plot='ts', display_format='streamlit')  

with col2:
    # Finalizar o modelo
    final_best = s.finalize_model(best)        
    st.write("**Modelo finalizado:**")
    st.write(final_best)

with col3:
    # Plotar previsões
    st.write("**Previsão com horizonte de 36 períodos:**")
    s.plot_model(final_best, plot='forecast', data_kwargs={'fh': 36}, display_format='streamlit')

# Exibindo previsões e métricas
col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1], gap='large')

with col1:
    predictions = s.predict_model(final_best, fh=36)     
    st.write("**Previsões:**")
    st.dataframe(predictions, use_container_width=True)
    
    # Botão para download das previsões
    csv = predictions.to_csv(index=False).encode('utf-8')
    st.download_button("Baixar Previsão",
                        data=csv, 
                        file_name="predictions.csv",
                        mime='text/csv',
                        key='download_button_previsao_auto')

with col2:
    st.write('Previsão 2025')
    st.metric(label='2025: Previsão e Variação %', value=2654064, delta=6.7)

with col3:
    st.write('Previsão 2026')
    st.metric(label='2026: Previsão e Variação %', value=2778932, delta=4.7)

with col4:
    st.write('Previsão 2027')
    st.metric(label='2027: Previsão e Variação %', value=2841797, delta=2.3)

with col5:
    st.write('Acuracidade dezembro/2024')
    st.metric(label='Dez 2024: Previsto', value=259110)
    st.metric(label='Dez 2024: Real e Acuracidade %', value=243751, delta=-6.3)


col1, col2=st.columns([1,1], gap='large')

with col1:
    with st.expander('Explicação Orthogonal Matching Pursuit w/ Cond. Deseasonalize & Detrending'):
        st.markdown('''O **Orthogonal Matching Pursuit (OMP)** é um modelo de aprendizado de máquina usado principalmente para tarefas de regressão e seleção de características. Quando aplicado à previsão de séries temporais, o modelo pode ser combinado com técnicas de **condicional deseasonalização e detrend**, o que melhora a capacidade do modelo em lidar com padrões sazonais e tendências nos dados antes de realizar a previsão. Vamos entender como funciona esse modelo de maneira geral e como ele se adapta à previsão de séries temporais.

### 1. **Orthogonal Matching Pursuit (OMP)**

**Orthogonal Matching Pursuit (OMP)** é um algoritmo de regressão que seleciona as variáveis mais importantes para o modelo de maneira iterativa. A ideia é aproximar a solução de um problema de regressão linear em um subespaço de variáveis de forma sequencial. O algoritmo constrói um modelo ajustando um vetor de pesos às variáveis selecionadas, de forma que o erro de previsão seja minimizado.

#### Como o OMP funciona:
- **Escolha inicial**: O OMP começa com um modelo que faz a previsão sem nenhuma variável. Ou seja, inicialmente ele não usa nenhuma das variáveis.
- **Seleção de características**: Em cada iteração, o algoritmo escolhe a variável que contribui mais para a redução do erro entre as previsões e os valores reais. A escolha é feita com base em um critério de "correlação" entre a variável e o erro residual (resíduos da previsão).
- **Ajuste do modelo**: Após selecionar a característica, o algoritmo ajusta os coeficientes das variáveis para que o modelo minimize o erro em relação aos dados.
- **Iteração**: Esse processo se repete até que o erro de previsão seja suficientemente pequeno ou até que o número máximo de iterações seja alcançado.

OMP é um tipo de **algoritmo de seleção de características** que tenta construir uma solução esparsa, ou seja, um modelo onde apenas algumas das características (ou variáveis) têm coeficientes diferentes de zero.

### 2. **Condicional Deseasonalização e Detrend**

A **deseasonalização** e o **detrend** são técnicas fundamentais em séries temporais para remover os padrões sazonais e as tendências, respectivamente.

- **Deseasonalização**: Refere-se ao processo de remover os efeitos sazonais (flutuações periódicas e previsíveis) da série temporal. As flutuações sazonais podem ser causadas por diferentes fatores, como estações do ano, feriados, ou outros ciclos regulares que afetam os dados.
  
  A deseasonalização pode ser feita subtraindo ou dividindo a série por um componente sazonal estimado. Isso permite que o modelo se concentre nas variações que não são sazonais.

- **Detrending**: Refere-se ao processo de remover a tendência de longo prazo (um movimento direcional crescente ou decrescente) de uma série temporal. A tendência pode ser linear ou não-linear, e sua remoção permite que o modelo foque nas flutuações residuais da série temporal.

  Para remover a tendência, pode-se subtrair uma linha de tendência (obtida por regressão linear, por exemplo) da série original.

### 3. **Combinação de OMP com Deseasonalização e Detrend**

Ao combinar o modelo **Orthogonal Matching Pursuit** com as técnicas de **condicional deseasonalização e detrend**, o processo de previsão de séries temporais passa por várias etapas, que podem ser descritas da seguinte maneira:

1. **Pré-processamento da Série Temporal**:
   - **Deseasonalização**: Remove os componentes sazonais, ou seja, os efeitos cíclicos regulares. Isso ajuda o modelo a não aprender variações sazonais, permitindo que ele se concentre em outras dinâmicas da série.
   - **Detrending**: Elimina a tendência de longo prazo, permitindo que o modelo se concentre em flutuações que não são causadas por uma tendência crescente ou decrescente.

2. **Aplicação do Modelo OMP**:
   - Após a deseasonalização e detrend, o modelo **Orthogonal Matching Pursuit** é aplicado à série de dados transformada. O OMP seleciona as características (variáveis) mais relevantes para o modelo, eliminando variáveis menos relevantes e ajustando os coeficientes das variáveis selecionadas para minimizar o erro de previsão.
   - O OMP é eficaz aqui porque, ao trabalhar com uma série temporal já ajustada para remover a sazonalidade e a tendência, ele pode identificar as relações subjacentes nas flutuações de curto prazo e aprender padrões sem ser influenciado por efeitos sazonais ou tendências de longo prazo.

3. **Previsão**:
   - Após a modelagem, as previsões podem ser feitas para o futuro usando o modelo ajustado.
   - As previsões geradas pelo modelo OMP podem então ser ajustadas de volta, adicionando novamente os componentes de sazonalidade e tendência para gerar previsões "no espaço original" da série temporal (não transformada).

### 4. **Vantagens do OMP com Deseasonalização e Detrend**:

- **Eficiência Computacional**: O OMP é relativamente simples e eficiente em termos de tempo de execução, o que é útil quando lidamos com séries temporais grandes e complexas.
  
- **Redução do Erro**: Ao remover a sazonalidade e a tendência dos dados antes de aplicar o modelo, o OMP pode ser mais eficiente em capturar as flutuações e os padrões reais nos dados. Isso melhora a precisão da previsão ao eliminar "ruído" causado por efeitos sazonais ou tendência.

- **Modelo Esparso**: O OMP tende a selecionar apenas um subconjunto das variáveis (ou características), o que pode ajudar a reduzir o risco de overfitting e melhorar a interpretação do modelo.

- **Adaptabilidade**: A combinação de OMP com deseasonalização e detrend pode ser aplicada a uma ampla gama de séries temporais, com diferentes padrões sazonais e tendências.

### 5. **Aplicação Prática**:

Este modelo é muito útil para séries temporais onde as flutuações sazonais e tendências são significativas e precisam ser removidas para que o modelo aprenda os padrões de curto prazo. Exemplos típicos de séries temporais que se beneficiariam dessa combinação incluem:
- **Vendas** de produtos que têm forte sazonalidade (como vendas de roupas de inverno, brinquedos no Natal, etc.).
- **Preços de ativos financeiros** (como ações, commodities) que podem ter uma tendência de longo prazo, mas também flutuações mais curtas e imprevisíveis.
- **Consumo de energia elétrica** que pode ter variações sazonais relacionadas ao clima e mudanças em longo prazo devido a fatores econômicos.

### 6. **Resumo**:

A técnica **Orthogonal Matching Pursuit com Condicional Deseasonalização e Detrend** aplicada à previsão de séries temporais é uma abordagem robusta que combina a eficiência de OMP na seleção de características com o pré-processamento das séries para remover padrões sazonais e tendências. Isso permite que o modelo aprenda melhor os padrões de curto prazo, melhorando a precisão das previsões e tornando-o uma ferramenta útil para tarefas de previsão em séries temporais com forte sazonalidade e tendência.
''', unsafe_allow_html=True)
   
with col2:
    with st.expander('**Análise dos Resultados**'):
        st.markdown('''Com base nos resultados apresentados para o modelo **Orthogonal Matching Pursuit w/ Cond. Deseasonalize & Detrending** (omp_cds_dt), podemos realizar a seguinte análise:

### Resultados:
- **MASE (Mean Absolute Scaled Error)**: 0.3564  
- **RMSSE (Root Mean Squared Scaled Error)**: 0.2680  
- **MAE (Mean Absolute Error)**: 10,906.5668  
- **RMSE (Root Mean Squared Error)**: 10,906.5668  
- **MAPE (Mean Absolute Percentage Error)**: 0.0451  
- **SMAPE (Symmetric Mean Absolute Percentage Error)**: 0.0463  
- **Tempo de Treinamento (TT)**: 0.0433 segundos  

### Análise dos Resultados:

1. **MASE (Mean Absolute Scaled Error)**:
   - O **MASE** de 0.3564 indica que o modelo tem um erro médio absoluto escalado de cerca de 35.64% em relação a uma previsão simples (como uma previsão com a média da série temporal).
   - **Interpretação**: Quanto menor o MASE, melhor o modelo. Nesse caso, o MASE sugere que o modelo está proporcionando uma melhoria significativa em relação a uma simples previsão média, mas ainda há espaço para otimização.

2. **RMSSE (Root Mean Squared Scaled Error)**:
   - O **RMSSE** de 0.2680 indica que a raiz quadrada do erro quadrático médio escalado também está abaixo de 1, o que é um bom sinal de que o modelo está ajustando-se bem à série temporal. Isso significa que, em média, o erro quadrático do modelo é 26.80% do erro quadrático de uma previsão de média.

3. **MAE (Mean Absolute Error)**:
   - O **MAE** de 10,906.5668 mostra que, em média, o erro absoluto do modelo para cada previsão é de aproximadamente 10,906 unidades (ou qualquer que seja a unidade da variável prevista).
   - **Interpretação**: Esse valor é relevante para entender a magnitude do erro em termos absolutos, mas, por si só, não diz muito sobre a qualidade do modelo, pois depende da escala dos dados.

4. **RMSE (Root Mean Squared Error)**:
   - O **RMSE** de 10,906.5668 é idêntico ao MAE, o que pode ser um sinal de que o modelo não apresenta grandes outliers, já que o RMSE é geralmente mais sensível a valores extremos do que o MAE. Isso sugere um desempenho equilibrado e sem grandes desvios.

5. **MAPE (Mean Absolute Percentage Error)**:
   - O **MAPE** de 0.0451 indica que o erro percentual médio é de 4.51%. Em geral, um MAPE abaixo de 10% é considerado bom, então esse valor sugere que o modelo está fazendo previsões razoavelmente precisas quando comparado aos valores reais.

6. **SMAPE (Symmetric Mean Absolute Percentage Error)**:
   - O **SMAPE** de 0.0463 (ou 4.63%) também é semelhante ao MAPE, indicando que a porcentagem de erro é bem controlada. O SMAPE é uma métrica mais robusta para séries temporais, pois lida de maneira equilibrada com os erros em valores próximos de zero, e esse valor também sugere um bom desempenho.

7. **Tempo de Treinamento (TT)**:
   - O **tempo de treinamento** foi de 0.0433 segundos, o que é extremamente rápido e eficiente. Isso é especialmente importante para grandes volumes de dados ou quando o modelo precisa ser treinado repetidamente.

### Conclusão:

Com base nesses resultados, o modelo **Orthogonal Matching Pursuit com Condicional Deseasonalização e Detrend (omp_cds_dt)** apresenta um bom desempenho na previsão da série temporal. O modelo conseguiu reduzir os erros de previsão com relação à média simples e gerou previsões com uma baixa margem de erro percentual. O tempo de treinamento também foi muito eficiente.

Em resumo:
- O modelo é eficaz em remover tendências e sazonalidades, e em fazer previsões com boa precisão.
- As métricas de erro sugerem que o modelo é robusto e oferece uma boa capacidade de previsão.
- A velocidade de treinamento é excelente, o que indica boa escalabilidade em caso de implementação em problemas maiores ou em tempo real.

Embora os resultados sejam positivos, seria interessante testar outras abordagens de modelagem ou ajustar os hiperparâmetros para melhorar ainda mais a precisão do modelo, se necessário.
''')

