# Módulo - Librosa

## Introdução
Este trabalho consiste na utilização de espectros de imagens geradas com a biblioteca  Librosa na linguagem Python e um arquivo .csv com dados  de coeficientes cepstrais de frequência de mel (CCFM), centróide espectral, taxa de cruzamento zero, frequência de croma e roll-off espectral.

## Objetivo 
O desenvolvimento de aplicações para o machine learning consiste em desenvolver algoritmos que possam trazer respostas inteligentes à problemas complexos. Pensando nisto este trabalho estará se baseando em um classificador para músicas em diferentes gêneros. A ideia é possibilitar a classificação em um cenário aleatorizado, sendo que a tarefa principal do classificador é mostrar uma acurácia satisfatória, que reconheça os gêneros escolhidos para o aprendizado de máquina, sendo jazz, música clássica, country, pop, rock e metal.
Processamento de áudio
Som é a propagação de uma frente de compressão mecânica ou onda mecânica; é uma onda longitudinal, que se propaga de forma circuncêntrica, apenas em meios materiais (que têm massa e elasticidade), como os sólidos, líquidos ou gasosos.
O processamento de Sinais é uma forma de se analisar ou alterar sinais fazendo uso da teoria fundamental, aplicações e algoritmos, e tem por objetivo extrair informações que sejam possíveis de utilização em alguma aplicação específica.
Os sons na forma digital possibilitam que sejam realizadas leituras e aanalises computacionais. Alguns formatos são mais conhecidos e utilizados como mp3, WMA (Windows Media Audio) e wav (arquivo de áudio em forma de onda).

## Librosa
A biblioteca Librosa é essencial para o tratamento de sinais de áudio é possui módulos que possibilitam a criação de imagens espectrais de sinais de áudio, e também pode extrair variáveis específicas para utilização em classificadores em formato .csv.

## Gerando arquivo de áudio
Quando é gerado um arquivo áudio é possível definir a taxa de amostragem do padrão (sr) em uma série temporal de áudio como uma matriz numpy. Pode-se então escolher o comportamento passando como parâmetro o valor da taxa de frequência de reamostragem desejada ou definindo como None, neste caso para desativar a reamostragem.
Exemplo de leitura para (a) 35,2 KHz e (b) None:
librosa.load (caminho_do_arquivo, sr = 35200)
librosa.load (caminho_do_arquivo, sr = None)
A taxa de amostragem é o número de amostras de áudio transportadas por segundo, e é medidas em Hz ou kHz.

## Espectrograma
É a maneira de se representar visualmente o espectro de frequências de som ou outros sinais, pois eles variam com o tempo. Os espectrogramas são conhecidos geralmente por resultados de exames de ultrassonografia e radiografias. Os dados representados em formato gráfico 3D são conhecidos como cascatas. Utiliza-se para matrizes em 2D o primeiro eixo para frequência e o segundo eixo para o tempo.


## Espectro de sinal de som  - Frequência x Tempo
### Extração de características
A extração de características consiste em aproveitar os recursos oferecidos na geração de sinal de áudio. Mas o mais importante neste processo de extração é aproveitar características que são importantes na resolução do problema definido para o classificador de gêneros musicais. Dentre as características estão : Zero Crossing Rate, Spectral Centroid, Spectral Rolloff, Mel-Frequency Cepstral Coefficients,  Chroma Frequencies.
A Taxa de Cruzamento Zero (Zero Crossing Rate) é a taxa de alteração de sinal ao longo de um sinal, ou seja, a taxa na qual o sinal muda de positivo para negativo ou para trás.
O Centróide Espectral (Spectral Centroid) indica onde o "centro de massa" de um som está localizado e é calculado como a média ponderada das frequências presentes no som.
O Rolloff Espectral (Spectral Rolloff) é uma medida da forma do sinal e representa a frequência abaixo da qual uma porcentagem especificada da energia espectral total.
Os Coeficientes Cepstrais de Frequência Mel (Mel-Frequency Cepstral Coefficients) (MFCCs) de um sinal são um pequeno conjunto de recursos (geralmente de 10 a 20) que descrevem de forma concisa a forma geral de um envelope espectral. 
Os recursos de Frequência de Croma (Chroma Frequencies) são uma representação interessante e poderosa para o áudio da música, na qual todo o espectro é projetado em 12 caixas, representando os 12 semitons distintos (ou croma) da oitava musical.


## Classificando as músicas em diferentes gêneros
### Conjunto de Dados
O conjunto de dados utilizado para este trabalho foi o GITZAN, este conjunto consiste na " Classificação de gênero de sinais de áudio " por G. Tzanetakis e P. Cook em IEEE Transactions on Audio and Speech Processing 2002 e artigo escrito por Parul Pandey que implementa uma guia de sequência para construção de um classificador com este objetivo.
O conjunto de dados possui 1000 faixas de áudio de 30 segundos. E está dividida em 10 pastas de acordo com o gênero musical, sendo blues, clássico, country, disco, hiphop, jazz, reggae, rock, metal e pop. E cada pasta contém 100 clipes de áudio.

### Pré-processo de dados
Para que os dados possam ser processados na linguagem Python é necessário converter os arquivos brutos do formato .au para o formato .wav, e para isto será utilizado o módulo SoX de código aberto.

sox input.au output.wav
Código para conversão .au para .wav

### Classificação
Para colocar em prática o processo de classificação é necessário extrair recursos significativos de áudio. Para que se possa classificar os clipes de áudio foram extraídos 5 recursos, Mel-Frequency Cepstral Coefficients, Spectral Centroid, Zero Crossing Rate, Chroma Frequencies, Spectral Roll-off. Todos os recursos são anexados a um arquivo .csv para que os algoritmos de classificação possam ser usados.

Tendo gerados os recursos .csv podemos usar algoritmos existentes para classificar as músicas em diferentes gêneros. Também podem ser utilizadas as imagens do espectrograma diretamente para classificação ou pode extrair os recursos e usar os modelos de classificação neles.

### Treinamento
O treinamento dos dados consiste na definição de grupos de dados que sejam de interesse da pesquisa. Para este problema foi utilizado a biblioteca Scikit-Learn que é uma ferramenta de Incorporação Estocástica de Vizinho distribuída em t.
O t-SNE [1] é uma ferramenta para visualizar dados de alta dimensão. Ele converte semelhanças entre pontos de dados em probabilidades conjuntas e tenta minimizar a divergência de Kullback-Leibler entre as probabilidades conjuntas da incorporação de baixa dimensão e os dados de alta dimensão. O t-SNE possui uma função de custo que não é convexa, ou seja, com diferentes inicializações, podemos obter resultados diferentes.

Foi definido para o parâmetro n_components (Dimensão do espaço incorporado) um valor int opcional igual a 2. Para a função fit_transform() os dados contidos na variável X_train que consiste num conjunto de treino gerado pela função train_test_split() que também faz parte da biblioteca Scikit-Learn. 
A plotagem abaixo foi gerada a partir da biblioteca Seaborn que consiste na geração de resultados em forma gráfica. Os dados abaixo foram plotados com uma paleta de 9 cores diferentes para cada grupo de dados encontrado.

### Plotagem com resultado de treinamento

Os classificadores utilizados neste problema foram definidos a partir da biblioteca Scikit-Learn, sendo eles DecisionTreeClassifier, KNeighborsClassifier, MLPClassifier, LogisticRegression, GaussianNB. Também foram o classificador por camadas layers() da biblioteca Keras.
Para os melhores classificadores foram utilizados os ensenble VotingClassifier(), SVC(), RandomForestClassifier(), BaggingClassifier() e o GradientBoostingRegressor() da biblioteca Scikit-Learn, que consiste em um método de agrupamento para obtenção de um melhor desempenho preditivo.

## Resultados
Para os algoritmos de classificação utilizados neste problema, os melhores resultados foram obtidos com LogisticRegression(), DecisionTree (), KNN(), sendo portanto aplicados ao ensenble VotingClassifier() com uma acurácia inferior ao ensemble SVM() que mostrou-se mais eficiente. O resultado mostrou-se com uma confiabilidade de 67% o que mostra que o classificador SVM deixou de acertar 33% dos gêneros musicais.
	

### Borda de decisão do classificador SVM

A informação de borda é utilizada inicialmente para separar regiões. E as propriedades espaciais e espectrais irão unir áreas com mesma textura. Ou seja, isto significa que as bordas identificadas possuem valores bem próximos para cada conjunto encontrado. E cada conjunto se divergem muito, um em relação ao outro, o que faz com que o classificador identifique um possível gênero.

 [14  0  2  0  0  2  0  0  0  1]
 [ 0 23  0  0  0  0  0  0  0  0]
 [ 0  0 12  5  1  2  0  0  0  1]
 [ 0  0  1 11  4  0  2  1  3  2]
 [ 0  0  0  1 14  0  0  0  2  1]
 [ 0  0  2  1  1 10  0  0  0  0]
 [ 1  0  0  1  1  0 18  0  0  0]
 [ 0  0  2  1  0  1  0 16  0  0]
 [ 0  0  1  0  1  1  0  1 11  0]
 [ 2  0  2  6  1  3  2  2  1  6]

Matriz de confusão do classificador SVM

A tabela acima mostra as frequências de classificação para cada classe do modelo. Sendo que a mesma apresenta na diagonal principal os valores encontrados corretamente para cada gênero musical e nas demais colunas os valores que mostram a quantidade de vezes que o classificador confundiu uma determinada classe com outra.

## Para maiores detalhes acesse: https://docs.google.com/document/d/1m5fVMmwS0OcIGwm71dEWjXa4QX_5-HDKQqSxweH8q_s/edit?usp=sharing


## Referências
SITE,seaborn.pydata.org/introduction.html , acessado em 28/11/2019.
SITE,scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html, acessado em 28/11/2019.
SITE,medium.com/data-hackers/como-criar-k-fold-cross-validation-na-m%C3%A3o-em-python-c0bb06074b6b, acessado em 29/11/2019.
SITE,medium.com/data-hackers/entendendo-o-que-%C3%A9-matriz-de-confus%C3%A3o-com-python-114e683ec509, acessado em 01/12/2019.
