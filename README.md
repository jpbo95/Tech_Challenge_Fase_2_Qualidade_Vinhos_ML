# Tech Challenge - Fase 2

## Predição da Qualidade de Vinhos por meio de Machine Learning

### 1. Introdução

A indústria vitivinícola utiliza diversas análises físico-químicas para avaliar as características dos vinhos. Entretanto, a avaliação da qualidade normalmente depende de especialistas, tornando o processo subjetivo.

Neste projeto será desenvolvido um modelo de Machine Learning capaz de estimar a qualidade de um vinho utilizando apenas suas propriedades físico-químicas.

Serão realizadas as etapas de análise exploratória dos dados (EDA), preparação da base, treinamento de diferentes algoritmos de regressão, avaliação dos results e comparação entre os modelos desenvolvidos.

### 2. Objetivo

Construir um modelo de Machine Learning capaz de prever a variável **quality** a partir das propriedades físico-químicas presentes no conjunto de dados.

Também serão avaliados diferentes algoritmos de regressão, identificando aquele que apresenta o melhor desempenho.

### 3. Dataset

Foi utilizado o conjunto de dados Wine Quality Dataset, disponível publicamente no Kaggle, contendo características físico-químicas de diferentes amostras de vinhos e sua respectiva classificação de qualidade. Cada amostra também possui uma nota de qualidade atribuída por especialistas, que será utilizada como variável alvo para o desenvolvimento de modelos de classificação.

Variável alvo:

- Qualidade

Variáveis preditoras:

- Acidez fixa
- Acidez volátil
- Ácido cítrico
- Açúcar residual
- Cloretos
- Dióxido de enxofre livre
- Dióxido de enxofre total
- Densidade
- pH
- Sulfatos
- Teor alcoólico

Quantidade de registros: 1.143

### 4. Tecnologias Utilizadas

- Python 3.11
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook

### 5. Estrutura do projeto
```
│
├── data/
├── notebooks/
├── src/
│   ├── transforms.py
│   ├── ml_models.py
│   ├── evaluate.py
│   └── plots.py
│
├── requirements.txt
└── README.md
```

### 6. Metodologia

O desenvolvimento do projeto foi dividido nas seguintes etapas:

1. Importação dos dados
2. Análise exploratória (EDA)
3. Tratamento e limpeza
4. Engenharia de atributos
5. Separação entre treino e teste
6. Padronização das variáveis
7. Treinamento dos modelos
8. Avaliação utilizando métricas de classificação
9. Comparação dos resultados

### 7. Modelos avaliados

- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine
- Logistic Regression

### 8. Como executar

Clone o repositório:

```bash
git clone ...
```

Crie um ambiente virtual:

```bash
python -m venv venv
```

Ative o ambiente:
```bash
venv/Scripts/Activate
```

Instale as dependências:
```bash
pip install -r requirements.txt
```
Execute o notebook principal.

### 9. Resultados

Ao final do projeto, fizemos uma comparação entre os modelos utilizando as métricas:

- Accuracy
- Precision
- Recall
- F1-Score

Além disso foram analisadas:

- Matriz de confusão
- Classification Report

### 10. Conclusão

A análise demonstrou que o algoritmo Random Forest apresentou o melhor desempenho entre os modelos avaliados, alcançando uma acurácia de 92,14%. Além do desempenho superior, o modelo permitiu identificar quais características físico-químicas exercem maior influência sobre a qualidade dos vinhos.

Os resultados indicam que as variáveis mais relevantes foram:

- Álcool
- Sulfatos
- Ácidez Volátil
- Ácido Cítrico
- Density

Essas informações podem auxiliar produtores e enólogos na priorização do controle das etapas do processo produtivo.


**Controle da fermentação**

O teor de álcool foi a variável mais importante para a classificação da qualidade dos vinhos.

Como o álcool é produzido durante a fermentação dos açúcares pelas leveduras, esse resultado sugere que o controle dessa etapa exerce influência significativa sobre a qualidade final do produto.

Na prática, fatores como:

- temperatura de fermentação;
- escolha da levedura;
- tempo de fermentação;
- teor inicial de açúcar da uva;

podem impactar diretamente a qualidade percebida do vinho.


**Controle microbiológico**

Os sulfatos e o dióxido de enxofre total aparecem entre as variáveis mais importantes.

Esses compostos estão relacionados à estabilidade microbiológica e à conservação do vinho, reduzindo a oxidação e o crescimento de microrganismos indesejáveis.

Esse resultado indica que o correto controle da adição desses compostos pode contribuir para manter as características sensoriais do produto ao longo do armazenamento.


**Redução da acidez volátil**

A acidez volátil aparece como a terceira variável mais importante.

Esse parâmetro normalmente está associado à presença de ácido acético, que em concentrações elevadas pode provocar aromas desagradáveis, semelhantes ao vinagre.

Dessa forma, controlar contaminações bacterianas e garantir boas condições durante a fermentação pode contribuir para a obtenção de vinhos de maior qualidade.


**Controle da densidade**

A densidade também apresentou elevada importância.

Essa variável está diretamente relacionada ao equilíbrio entre álcool e açúcares dissolvidos, sendo frequentemente utilizada para acompanhar a evolução da fermentação.

Seu destaque confirma que o monitoramento da fermentação é uma etapa crítica do processo produtivo.


**Variáveis de menor impacto**

Por outro lado, características como:

- açúcar residual;
- dióxido de enxofre livre;
- pH

apresentaram menor contribuição individual para o modelo.

Isso não significa que sejam irrelevantes para a produção do vinho, mas indica que, considerando este conjunto de dados, possuem menor poder discriminatório para diferenciar os níveis de qualidade quando comparadas às demais variáveis.


**Considerações finais**

Os resultados obtidos demonstram que técnicas de Machine Learning podem ser utilizadas como ferramenta de apoio ao controle de qualidade na indústria vinícola. A identificação das variáveis mais relevantes permite direcionar esforços para o monitoramento dos fatores que apresentam maior impacto sobre a qualidade final do produto, auxiliando na tomada de decisão durante o processo de fabricação.

Embora o modelo não substitua a avaliação sensorial realizada por especialistas, ele pode atuar como um sistema de apoio à decisão, permitindo identificar precocemente lotes com maior probabilidade de apresentar qualidade inferior. Isso possibilita ações corretivas ainda durante a produção, reduzindo desperdícios, aumentando a padronização e contribuindo para a melhoria contínua do processo produtivo.

### Autores:

- Fabio Silveira Beneti

- Jéssica Vieira de Souza

- João Pedro Brito Oliveira

- Mateus Cabral Gama Oliveira

- Matheus Barros de Santana

**Projeto desenvolvido para o Tech Challenge – Fase 2 – Pós-Tech Data Analytics – FIAP**



