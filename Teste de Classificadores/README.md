# Python - Teste de Classificadores para Gêneros Musicais

## Introdução
Este trabalho apresenta como corpo resultados obtidos com a elaboração de um algoritmo desenvolvido em linguagem Python, que apresenta resultados de um conjunto de classificadores que tem como finalidade analisar dados numéricos e aprendizado com imagens espectrais de áudio oriundas de experimentos com diferentes gêneros musicais, buscando como resultado de importância a busca por um classificador ou uma combinação de classificadores que mostrarem uma melhor acurácia.

## Classificadores
Os algoritmos classificadores também conhecidos como algoritmos de aprendizado de máquina (machine learning) que são uma ferramenta voltada a aplicações que buscam padrões para identificação por meio de inteligência artificial, junto a resultados probabilísticos.
Para os testes realizados neste trabalho foram utilizados os algoritmos Logistic Regression(), DecisionTreeClassifier(), KNeighborsClassifier(), MLPClassifier(), GaussianNB(), RandomForestClassifier(), VotingClassifier(), BaggingClassifier() e o GradientBoostingRegression().

## GridSearchCV()
Este algoritmo realiza pesquisa exaustiva sobre os valores de parâmetros especificados para um estimador. O GridSearchCV implementa um método de "ajuste" e "pontuação". Também implementa “prever”, “prever_proba”, “função de decisão”, “transformar” e “transformação inversa” se eles forem implementados no estimador usado.
Os parâmetros do estimador usado para aplicar esses métodos são otimizados pela pesquisa de grade validada cruzadamente sobre uma grade de parâmetros.

## Logistic Regression()
Apesar da palavra Regressão em Regressão Logística, a Regressão Logística é um algoritmo de aprendizado de máquina supervisionado usado na classificação binária. Digo binário porque uma das limitações da regressão logística é o fato de que ele só pode categorizar dados com duas classes distintas. Em um nível alto, a Regressão Logística ajusta uma linha a um conjunto de dados e retorna a probabilidade de uma nova amostra pertencer a uma das duas classes, de acordo com sua localização em relação à linha.

## DecisionTreeClassifier()
Uma árvore de decisão é uma estrutura de árvore do tipo fluxograma em que um nó interno representa um recurso (ou atributo), o ramo representa uma regra de decisão e cada nó folha representa o resultado. O nó mais alto em uma árvore de decisão é conhecido como nó raiz. Aprende a particionar com base no valor do atributo. Ele particiona a árvore de maneira recursiva, chamada de particionamento recursivo. Essa estrutura semelhante ao fluxograma ajuda na tomada de decisão. É a visualização como um diagrama de fluxograma que imita facilmente o pensamento no nível humano. É por isso que as árvores de decisão são fáceis de entender e interpretar.

## KNeighborsClassifier()
O KNN é usado em diversas aplicações, como finanças, saúde, ciência política, detecção de manuscrito, reconhecimento de imagem e vídeo. Nas classificações de crédito, os institutos financeiros farão previsão para a classificação de crédito dos clientes. No desembolso de empréstimos, os institutos bancários farão previsão se o empréstimo é seguro ou arriscado. Na ciência política, a classificação de potenciais eleitores em duas classes votará ou não. Algoritmo KNN usado para problemas de classificação e regressão. Algoritmo KNN baseado na abordagem de similaridade de recursos.

## MLPClassifier()
Uma comparação de valores diferentes para o parâmetro de regularização “alpha” em conjuntos de dados sintéticos. O gráfico mostra que diferentes alfas produzem diferentes funções de decisão. Alfa é um parâmetro para o termo de regularização, também conhecido como termo de penalidade, que combate o overfitting ao restringir o tamanho dos pesos. O aumento de alfa pode corrigir alta variância (um sinal de overfitting), encorajando pesos menores, resultando em um gráfico de limite de decisão que aparece com curvaturas menores. Da mesma forma, o decréscimo de alfa pode corrigir um alto viés (um sinal de mau preparo) encorajando pesos maiores, resultando potencialmente em um limite de decisão mais complicado.

## GaussianNB()
O classificador Naive Bayes é um algoritmo simples e poderoso para a  tarefa de classificação . Mesmo se estivermos trabalhando em um conjunto de dados com milhões de registros com alguns atributos, é recomendável tentar a abordagem Naive Bayes. O classificador Naive Bayes oferece ótimos resultados quando o usamos para análise de dados textuais. Tais como processamento de linguagem natural.

## RandomForestClassifier()
As florestas aleatórias são um algoritmo de aprendizado supervisionado. Pode ser usado para classificação e regressão. É também o algoritmo mais flexível e fácil de usar. Uma floresta é composta de árvores. Diz-se que quanto mais árvores houver, mais robusta será a floresta. As florestas aleatórias criam árvores de decisão em amostras de dados selecionadas aleatoriamente, obtêm previsões de cada árvore e selecionam a melhor solução por meio de votação. Ele também fornece um bom indicador da importância do recurso.

## VotingClassifier()
Um classificador de votação é um método de aprendizado de conjunto e é um tipo de wrapper que contém diferentes classificadores de aprendizado de máquina para classificar os dados com votação combinada. Existem métodos de votação 'hard / maioria' e 'soft' para tomar uma decisão em relação à classe-alvo. A votação definitiva decide de acordo com o número de votos que é a maioria ganha. Na votação branda, podemos definir o valor do peso para dar mais prioridades a determinados classificadores de acordo com seu desempenho.

## BaggingClassifier()
Um classificador de votação é um método de aprendizado de conjunto e é um tipo de wrapper que contém diferentes classificadores de aprendizado de máquina para classificar os dados com votação combinada. Existem métodos de votação 'hard / maioria' e 'soft' para tomar uma decisão em relação à classe-alvo. A votação definitiva decide de acordo com o número de votos que é a maioria ganha. Na votação branda, podemos definir o valor do peso para dar mais prioridades a determinados classificadores de acordo com seu desempenho.

## GradientBoostingRegressor()
Regressores de reforço de gradiente (GBR) são modelos de regressores de árvore de decisão de conjunto. Neste trabalho, mostraremos como preparar um modelo GBR. Construiremos um modelo para estimar o gênero musical de vários dados numéricos. 

## Formalização do código
A seguir estão descritas as definições para cada função implementada, sendo SVM e MLP uma combinação de classificadores que serão testadas pelo VotingClassifier().
```ruby
# Define single classifiers

lr = LogisticRegression()
dt =  DecisionTreeClassifier(criterion='entropy')
knn = KNeighborsClassifier(n_neighbors=3)
mlp = MLPClassifier(solver='sgd', early_stopping=True, hidden_layer_sizes=(100), activation='logistic', batch_size=100, max_iter=10000, learning_rate_init=0.1, momentum=0.2, tol=1e-10, random_state= rng)
nb = GaussianNB(var_smoothing=1e-09)
 
# Define ensembles
svm = SVC(gamma='scale', kernel='rbf', probability=True)
rf = RandomForestClassifier(n_estimators=100, random_state = None)
cb = VotingClassifier(estimators=[('SVM', svm), ('MLP', mlp)], voting='soft')
bg  =  BaggingClassifier( knn, max_samples = 0.5 ,  max_features = 0.5 )
gbr = GradientBoostingRegressor(n_estimators=100,learning_rate=0.1)
 
# parameters for SVM
parameters = [
  {'C': [0.1, 0.5, 1, 10, 100, 500, 1000], 'kernel': ['poly']},
  {'C': [0.1, 0.5, 1, 10, 100, 500, 1000], 'gamma': [0.1, 0.001, 0.0001, 0.00001], 'kernel': ['rbf']},
]
svm = GridSearchCV(svm, parameters, scoring = 'accuracy', cv=10, iid=False)
 
titles = ['LogisticRegression', 'DecisionTree', 'KNN', 'NaiveBayes', 'MLP', 'SVM', 'RF', 'SVM+MLP', 'BG', 'GBR']
methods = [lr, dt, knn, nb, mlp, svm, rf, cb, bg, gbr]
thods = [lr, dt, knn, nb, mlp, svm, rf, cb, bg, gbr]
``` 

##Resultados
Abaixo estão representados os resultados de acordo com a acurácia de cada classificador, bem como as combinações do MLP com SVM e uma combinação com todos os classificadores, por meio do VotingClassifier().

| Classificador               | Acurácia |
|-----------------------------|----------|
| LogisticRegression()        | 0.6034   |
| DecisionTree()              | 0.47     |
| KNeighborsClassifier()      | 0.6267   |
| GaussianNB()                | 0.41     |
| MLPClassifier()             | 0.4935   |
| SVC()                       | 0.6833   |
| RandomForestClassifier()    | 0.6267   |
| VotingClassifier(MLP + SVM) | 0.6433   |
| VotingClassifier(ALL)       | 0.6034   |
| BaggingClassifier()         | 0.61     |
| GradientBoostingRegressor() | 0.319    |
