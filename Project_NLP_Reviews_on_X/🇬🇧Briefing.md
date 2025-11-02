
### 🇬🇧 Project Statement

**Sentiment Analysis with Naive-Bayes and N-grams**

Deliverable
`.ipynb` file with all cells already executed
Modality
Individual

Objective
Build and evaluate a Machine Learning model (Multinomial Naive-Bayes) to classify reviews into three sentiment categories: Negative (0), Neutral (1), and Positive (2).

Dataset

- File: `B2W-Reviews01.csv`

Notebook Workflow (9 Steps)
Your notebook should be clear, well documented (using Markdown), and run sequentially.


| Section | Instructions \& Requirements |
| :-- | :-- |
| 1 Preparation | Import the necessary libraries (pandas, nltk, sklearn, seaborn, matplotlib). Download NLTK resources (punkt, stopwords). |
| 2 Loading \& Cleaning | Load the DataFrame (review_text, overall_rating). Remove rows with null values (dropna()). Show total remaining reviews. |
| 3 Class Definition | Create the mapping function: 1 → 0 (Negative), 2, 3, 4 → 1 (Neutral), 5 → 2 (Positive). Apply and display the new class distribution. |
| 4 Pre-processing (NLP) | Create and apply preprocessText function: remove punctuation, convert to lowercase, tokenize, remove stopwords (Portuguese) and digits. Generate the texto_pre column. |
| 5 N-gram Vectorization | Instantiate CountVectorizer with ngram_range=(1,3). Fit/transform texto_pre to create the X feature matrix. Print X.shape. |
| 6 Data Split | Use train_test_split (20% for test, random_state=42) to split data. Print the shapes of train and test sets. |
| 7 Naive-Bayes Training | Instantiate and train (fit) the MultinomialNB() model. Print train and test accuracy. |
| 8 Complete Evaluation | Generate classification_report. Plot confusion_matrix as a heatmap (seaborn) to visualize class performance. |
| 9 Manual Test (Demo) | Create and run an interactive function (input()) that processes user text, vectorizes, predicts the class, and displays the final result with confidence level (predict_proba). |

**Focus on Assessment (Creativity and Documentation)**

- Documentation: Each section should have a brief Markdown header explaining the purpose of the step.
- Analysis: Include comments (in Markdown or code) about lost data volume (Step 2), class balancing (Step 3), and interpretation of classification_report (Step 8).
- Creative Test: Create 5 distinct example sentences to demonstrate model operation and limits in Step 9.

---

### 🇧🇷 Enunciado do Projeto

**Análise de Sentimento com Naive-Bayes e N-grams**

Entrega
Arquivo `.ipynb` com células já executadas
Modalidade
Individual

Objetivo
Construir e avaliar um modelo de Machine Learning (Naive-Bayes Multinomial) para classificar reviews em três categorias de sentimento: Negativo (0), Neutro (1) e Positivo (2).

Dataset

- Arquivo: `B2W-Reviews01.csv`

Roteiro do Notebook (9 Etapas)
Seu notebook deve ser claro, bem documentado (usando Markdown) e rodar sequencialmente.


| Seção | Instruções e Requisitos |
| :-- | :-- |
| 1 Preparação do Ambiente | Importar as bibliotecas necessárias (pandas, nltk, sklearn, seaborn, matplotlib). Baixar recursos NLTK (punkt, stopwords). |
| 2 Carregamento e Limpeza | Carregar o DataFrame (review_text, overall_rating). Remover linhas com valores nulos (dropna()). Exibir o total de reviews restantes. |
| 3 Definição das Classes | Criar a função de mapeamento de notas para 3 classes: 1 → 0 (Negativo), 2, 3, 4 → 1 (Neutro), 5 → 2 (Positivo). Aplicar e exibir a distribuição das novas classes. |
| 4 Pré-processamento (NLP) | Criar e aplicar a função preprocessText para: remover pontuação, converter para minúsculas, tokenizar, remover stopwords (Português) e dígitos. Gerar a coluna texto_pre. |
| 5 Vetorização N-grams | Instanciar CountVectorizer com ngram_range=(1,3). Ajustar/Transformar texto_pre para criar a matriz de recursos X. Imprimir o X.shape. |
| 6 Divisão dos Dados | Usar train_test_split (20% para teste, random_state=42) para separar os dados. Imprimir os shapes dos conjuntos de treino e teste. |
| 7 Treinamento Naive-Bayes | Instanciar e treinar (fit) o modelo MultinomialNB(). Imprimir a acurácia de treino e teste. |
| 8 Avaliação Completa | Gerar o classification_report. Plotar o confusion_matrix como um heatmap (seaborn) para visualização da performance por classe. |
| 9 Teste Manual (Demo) | Criar e rodar uma função interativa (input()) que: processa o texto do usuário, vetoriza, prevê a classe e exibe o resultado final com o nível de confiança (predict_proba). |

**Foco na Avaliação (Criatividade e Documentação)**

- Documentação: Cada seção deve ter um breve cabeçalho em Markdown explicando o propósito da etapa.
- Análise: Inclua comentários (em Markdown ou código) sobre o volume de dados perdido (Etapa 2), o balanceamento das classes (Etapa 3), e a interpretação dos resultados do classification_report (Etapa 8).
- Teste Criativo: Crie 5 exemplos de frases distintas para demonstrar o funcionamento e os limites do modelo na Etapa 9.



