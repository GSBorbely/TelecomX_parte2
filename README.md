# :floppy_disk: **Telecom X parte 2 - Análise de Evasão de Clientes**

A Telecom está passando por problemas no grande aumento de evasões de clientes e este projeto teve como objetivo demonstrar maneiras de evitar estas evasões (churn) buscando formas de entender quais foram os motivos que influenciaram esta decisão. Nesta segunda parte do desafio fomos promovidos a equipe de Machine Learning da empresa e somos responsáveis na criação de modelos preditivos para analisar os motivos das evasões.
Neste projeto foi utilizada a linguagem de programação Python e suas bibliotecas para analisar os dados, extrair as informações e aplicá-las nas tomadas de decisões. 

## ✔ Tecnologias utilizadas
- ``Python``
- ``Pandas``
- ``Numpy``
- ``Seaborn``
- ``Matplotlib ``
- ``Sklearn (MinMaxScaler, Train Test Split,RandomForestClassifier, LogisticRegression, classification_report, confusion_matrix, StandardScaler, accuracy_score) ``
- ``Jupyter Notebook pelo Google Colab ``,

# :clipboard: **Extração dos Dados**
Os dados foram extraidos através de uma API em formato JSON. Os dados utilizados foram os dados do último desafio, que foram tratados e normalizados para a utilização no treinamento dos modelos. 
## Remoção de colunas irrelevantes e transformação numérica.
Algumas colunas categóricas foram removidas para uma facilitação da análise; '['phone_MultipleLines', 'internet_InternetService',
'internet_OnlineSecurity', 'internet_OnlineBackup',
'internet_DeviceProtection', 'internet_TechSupport',
'internet_StreamingTV', 'internet_StreamingMovies', 'account_Contract',
'account_PaymentMethod'].

Após a remoção das irrelevantes, a restantes foram transformadas em numéricas através do get_dummies.

## Proporção de evasões e contagem de clientes
<img width="1183" height="484" alt="image" src="https://github.com/user-attachments/assets/cb3f6403-1a5d-4216-b611-39206bba01c3" />

O resultado demonstra os seguintes dados:

Contagem de Clientes:

- Clientes ativos (0.0): 5.174

- Clientes que evadiram (1.0): 1.869

Proporção:

- 73,46% dos clientes permaneceram.

- 26,54% dos clientes evadiram.
  
## Normalização ou Padronização
Foi utilizado o  MinMaxScaler para padronizar os dados (Min de 0 e máx de 1 para todas as colunas)

## Análise de Correlação
Com a análise da correlação entre as variavéis obteve-se um bom insight, demonstrando os seguintes resultados:

<img width="1492" height="1383" alt="image" src="https://github.com/user-attachments/assets/86a9cb68-602f-4b9f-9235-f2479604aa2a" />

Principais destaques do gráfico:
- customer_tenure -0.35 Clientes antigos têm menor chance de sair

- account_Charges.Total -0.20 Gastos totais maiores tendem a estar associados à permanência

- account_Charges.Daily/Monthly +0.19 Leve correlação positiva com churn

- customer_paperless_binario +0.19 Clientes com fatura digital tendem a sair mais

- customer_dependents_binario -0.16 Clientes com dependentes tendem a permanecer

- customer_SeniorCitizen +0.15 Idosos têm leve tendência a evadir

- customer_partner_binario -0.15 Quem tem parceiro tende a permanecer

## Separação em Teste e Treino e treinamento dos modelos 
Os dados foram dividos em 30/70, sendo 30% para testes e 70% para treino com o train test split

Para os modelos, foi utilizado a Regressão Logistica e o Random Forest. O primeiro modelo é o mais comum para se ter um primeiro insight, sendo sensivel ao tratamento dos dados e por este motivo resolvi escolhe-lo. Com esta análise, pude perceber que o modelo funcionou bem em apresentar os clientes que não evadiram, mas obteve dificuldade em apresentar os que evadiram. Já o Random Forestm pude utilizar os dados sem a necessidade de tratá-los e pude ter uma visão mais complexa dos dados, o que é uma de suas caracterisiticas. Desta forma identifiquei padrões mais relevantes para o resultado. 

<img width="537" height="479" alt="image" src="https://github.com/user-attachments/assets/36f14815-659a-4964-8409-97e204856dd3" />

- 1413 clientes foram corretamente classificados como não evadidos (verdadeiros negativos).

- 269 clientes foram corretamente identificados como evadidos (verdadeiros positivos).

- 136 clientes que permaneceram foram classificados como se tivessem evadido (falsos positivos).

- 292 clientes que evadiram foram classificados como se tivessem permanecido (falsos negativos).

- Precisão da classe 1 (evadidos): 66% dos clientes que o modelo previu como evadidos realmente evadiram.

- Recall da classe 1: o modelo identificou corretamente apenas 48% dos clientes que realmente evadiram (mais da metade passaram despercebidos).

- Acurácia geral: 80% — ou seja, 8 em cada 10 clientes foram classificados corretamente.

- F1-score da classe 1 (churn): apenas 0.56 — o modelo ainda está com dificuldade em capturar corretamente quem vai evadir.

<img width="537" height="479" alt="image" src="https://github.com/user-attachments/assets/c81fba14-a578-421b-815e-69cf79e2accb" />

- Verdadeiros Negativos (TN): 1357 Clientes que não evadiram e foram corretamente classificados como tal.

- Falsos Positivos (FP): 197 Clientes que não evadiram, mas o modelo previu que evadiriam.

- Falsos Negativos (FN): 281 Clientes que evadiram, mas o modelo previu que não evadiriam.

- Verdadeiros Positivos (TP): 275 Clientes que evadiram e foram corretamente identificados.

- Precisão da classe 1 (evadidos): Quando o modelo prevê que um cliente vai evadir, ele acerta 58% das vezes.

- Recall da classe 1: O modelo identificou corretamente 49% dos clientes que realmente evadiram.

- F1-score da classe 1: 0.54 — mostra que o desempenho nessa classe ainda é moderado.

- Acurácia geral: 77% — o modelo acerta 77 a cada 100 clientes no total.

## Underfitting ou Overfitting
O modelo de Regressão Logisitica apresentou acurácia parecida do treino e teste, ou seja, apresentou nenhum  ou quase nenhum overfitting.
Já no modelo Randon Forest, o modelo aprendeu perfeitamente os dados de treino, apresentando quase 100% de acurácia, indicando Overfitting. Para reduzi-lo, foi recomendado o uso do parâmetro max-depth para limitar a profundida máxima das árvores, e o Min-samples_split para limitar o minimo de amostras para dividir um nó, e/ou o Min-samples_leaf. Após as mudanças, o modelo apresentou uma melhora com a redução da acurácia dos dados de treino (85,5%), ou seja, o modelo parou de decorar os dados.

## Análise de Importância das Variáveis
Regressão Logistica:

<img width="1053" height="556" alt="image" src="https://github.com/user-attachments/assets/091ad971-b29a-4d34-8e00-d07ae8c954b9" />

Random Forest:

<img width="984" height="784" alt="image" src="https://github.com/user-attachments/assets/61fd4ca7-859f-4691-ac5b-7b5b1574df14" />

Conclusão e relatório final
Após a análise dos modelos obteve-se as seguintes conclusões e recomendações:

Tempo de contrato (Customer Tenure) - Um dos principais motivos de evasões dos clientes, tendo destaque nas duas análises dos dois modelos apresentados. Clientes com contratos mais curtos tendem a cancelar mais o serviço, principalmente os que possuem menos de 30 meses de adesão ao serviço. Neste caso o principal objetivo da Telecom deve ser o incentivo ao maior tempo de adesão logo no inicio do contrato, com preços que sejam atrativos nos tipos de contrato mais longos. Desta forma, haverá um grande diminuição nas evasões sendo que o este tipo de contrato apresenta uma maior porcentagem de retenção de clientes. A estratégia de marketing deve ser pontual, apresentando claramente as vantagens de se obter um contrato mais longo.

<img width="851" height="635" alt="image" src="https://github.com/user-attachments/assets/abd999de-0708-49bb-9776-81484213ca61" />

*Tipo do Contrato (Daily, Monthly e Total) * - Os clientes que pagam o serviço por dia e por mês são mais propensos a cancelar o serviço, relacionando-se com o valor cobrado que é a principal influência. Recomenda-se o estimulo a uma contratação de serviço anual, que apresenta uma maior retenção de clientes em relação aos outros tipos de contrato. As mensalidades mensais merecem uma grande atenção, e exige atenção em relação ao preço cobrado sendo necessário um reajuste no preço, e descontos que sejam mais atrativos aos clientes.

<img width="870" height="635" alt="image" src="https://github.com/user-attachments/assets/fa254949-8881-4712-810e-5c0eb2970f30" />

Cobrança Eletrônica por cheque - Outro fator de grande importância. A análise demonstra que grande parte dos que aderem a cobrança eletrônica tendem a cancelar o serviço mais frequentemente. Uma alternativa essencial para diminuir estas evasões é haver uma atenção maior em relação ao preço das faturas, aplicando pesquisas de satisfação relacionadas a este ponto e quais exatamentes são as insatisfações dos clientes. Outros tipos de cobrança devem ser recomendadas e incentivadas através de publicidade. A praticidade em um ponto tão importante como o pagamento das faturas deve ser prioridade, recomendando-se alternativas ao cheque eletrônico, como o pagamento por PIX, Cartão de Crédito, Cartão de Débito ou Débito automático.

 ##  Acesso ao projeto
Você pode [acessar o código do projeto](https://github.com/GSBorbely/TelecomX_parte2/blob/main/TelecomX_parte2.ipynb)








