---
layout: single
title:  "AutoML é útil, mas não vai fazer mágica em cima de seus problemas"
author: Weslley
date:   2020-05-29
categories: data-science machine-learning automl

excerpt: "Neste artigo vamos analisar alguns pontos sobre como bibliotecas de AutoML podem te ajudar, o nível de automação que elas oferecem e a curva de aprendizado para utilizar um exemplo (auto-sklearn)"
header:
  overlay_image: /assets/images/matrix_background.jpg
  overlay_filter: 0.7
---

# _Highlights_
- As ferramentas de AutoML parecem ter grande potencial de contribuição no cotidiano de um time de _Data Science_/_Machine Learning_, porém nas etapas mais adiantadas do processo de desenvolvimento e _realease_ de modelos;
- Etapas de entendimento do problema e engenharia de _features_ são pouco impactados por este tipo de ferramentas;
- Analisando o `auto-sklearn`, o processo automatizado tem umas carga de aprendizado bastante grande e exige um certo processo exploratório inicial para redução do espaço de busca de algoritmos e parâmetros para facilitar uma segunda etapa de otimização de hiperparâmetros propriamente dito;
- O código usado na exploração do `auto-sklearn` está disponível em um [repositório público](https://github.com/wespatrocinio/learning_automl/) na minha conta do GitHub;

# O artigo em si

Nos últimos meses tenho me dedicado à identificar e otimizar o cotidiano de times de _Machine Learning_ & AI com processos, políticas e novas tecnologias. Um dos pontos que tenho avaliado em mais detalhes é a utilização de bibliotecas de _automated machine learning_ (AutoML) para auxiliar o processo de encontrar o melhor modelo possível em desenvolvimento. Após algumas análises, decidi compartilhar algumas de minhas impressões neste artigo.

>"_I have a dream that one day my product is going to be supplied by self-assembled machine learning models_." (Algum C-level de alguma empresa anunciando que agora são _AI-first_ e que serão referência em AI nos próximos 2 anos).

## O que será considerado como um modelo neste artigo?

Na minha perspectiva (e acredito que não só minha), um modelo de _Machine Learning_ é composto por três partes fundamentais. Já alerto que simplificarei bastante a descrição das partes para facilitar o entendimento de quem não tem muito vocabulário deste contexto.

1. **_Dataset_**, que contém os dados com todas as _features_ e registros que serão utilizados para o treinamento e validação do modelo, independente do fato de ser supervisionado ou não;
2. **Algoritmo/Ensemble**, que é a ferramenta que será usada para encontrar determinados padrões nos dados do _dataset_ e ser capaz de utilizar tais padrões para processar novos registros que sejam requisitados para o modelo e realizar uma predição (classificação, regressão, _pattern recognition_, etc.).
3. **Hiper-parâmetros**: os parâmetros a serem usados pelo algoritmo e que serão otimizados (também chamado de _tunning_) de forma a obter maior capacidade de predição de um algoritmo sobre um determinado conjunto de dados. Alguns pontos importantes são:
    - Existem parâmetros tanto do algoritmo quanto do _dataset_ (por exemplo, balanço entre classes positivas e negativas em um caso de aprendizado supervisionado) e de algoritmo (por exemplo, a profundidade máxima de uma árvore de decisão);
    - Um mesmo algoritmo aplicado a dois _datasets_ diferentes pode (e provavelmente irá) requerer diferentes valores de parâmetros para alcançar a performance ótima;

![](/assets/images/ml-hyperparameter.png){: .align-center}
*[Diagrama simplificado sobre as etapas da criação de um modelo de ML](https://cambridgecoding.wordpress.com/2016/04/03/scanning-hyperspace-how-to-tune-machine-learning-models/)*
{: .text-center}

## Qual a minha compreensão sobre AutoML neste momento?

Ao explorar algumas bibliotecas de AutoML como o [auto-sklearn](https://automl.github.io/auto-sklearn/master/#), [H2O](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html), [Google Cloud AutoML](https://cloud.google.com/automl?hl=pt-br) e outras, identifiquei, de maneira geral, o seguinte comportamento em todas elas:
- oferecem funcionalidades de Engenharia de _Features_ que aplicam novas representações de um mesmo dado (por exemplo, fazer [_One Hot Encoding_](https://en.wikipedia.org/wiki/One-hot) de uma _feature_ categórica) mas não abordam muito bem a relevância delas;
- oferecem um grande portfolio de algoritmos que serão utilizados na busca pelo melhor modelo;
- permitem customizações de métricas e de algoritmos para permitir que os usuários utilizem a ferramenta de AutoML como uma plataforma de otimização que pode ser encaixada nas particularidades de uma empresa;

Dessa forma, as ferramentas que explorei não agregam muito valor à etapa de engenharia de _features_, com foco explícito no _tunning_ de algoritmos e hiper-parâmetros. Ao meu ver, isso limita bastante a magnitude do impacto que este tipo de ferramenta tem na automação do processo de criação de um modelo. Eis alguns pontos que sustentam esta minha opinião:
- Por mais que muitos acreditem que um modelo de ML é um _data monster_ que vai processar milhares de _features_ com dados de usuários dos últimos 10 anos e se tornar um oráculo das predições, não é o que acontece de fato. A etapa de tratamento de dados, construção de _features_ e identificação das mais relevantes compõe os maiores desafios na construção de modelos;
- Alguns algoritmos possuem premissas as quais os dados precisam respeitar para que sejam aplicados adequadamente (por exemplo, o [LDA](https://en.wikipedia.org/wiki/Linear_discriminant_analysis) assume que os dados tenham distribuição normal e que ambas as classes possuam uma mesma matriz de covariância). A automação apenas do processo de _tunning_ pode contribuir com a negligência deste tipo de validação;

## Um breve exemplo: algoritmos _vanilla_ _VS_ `auto-sklearn`

Para exemplificar de maneira mais tangível o valor percebido durante a minha exploração das bibliotecas, eu mostrarei um exemplo simples que construi para explorar como o `auto-sklearn` funciona e o quanto ele contribuiria para um modelo construído sobre dados que já conheço. Portanto, eu tomei como base um [modelo](https://github.com/wespatrocinio/music_genre_classification/blob/master/notebooks/music_genre_classification.ipynb) que fiz alguns anos atrás, para fins didáticos, para prever o gênero de uma música a partir de sua letra. Sendo assim, usei os mesmos dados e gerei as mesmas _features_ ([`tf-idf`](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) dos 200 _tokens_ mais relevantes) para treinar um modelo. Neste artigo eu não entrarei em detalhes sobre a geração destas _features_.


### _Vanilla Decision Tree_

Para ter um _baseline_ simples, vou criar um modelo usando [Árvore de Decisão](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) (_Decision Tree_) com os parâmetros padrão do `scikit-learn`.

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(train_features, train_target)
predicted_target = model.predict(test_features)
print("Accuracy score: ", sklearn.metrics.accuracy_score(test_target, predicted_target))
```

O resultado impresso por este trecho de código é:
```
Accuracy score: 0.6412903225806451
```

Sendo assim, um modelo criado sobre este conjunto de _features_ usando uma árvore de decisão "padrão" gerou um modelo com **64,1%** de [acurácia](https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification).

### _Vanilla Random Forests_

Da mesma forma que usei uma árvore de decisão sem otimização de hiper-parâmetros como _baseline_, farei o mesmo com [_Random Forests_](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(train_features, train_target)
predicted_target = model.predict(test_features)
print("Accuracy score: ", sklearn.metrics.accuracy_score(test_target, predicted_target))
```

O resultado impresso por este trecho de código é:
```
Accuracy score: 0.7896774193548387
```

A acurácia obtida por este modelo foi de **79,0%**, consideravelmente maior que o modelo _vanilla_ de Árvore de Decisão.

### `auto-sklearn`

O `auto-sklearn` é uma biblioteca construída em cima do [`scikit-learn`](https://scikit-learn.org/stable/) e que ganhou relevância ao ser [apresentado no NIPS](https://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning) e também ao [vencer um desafio de AutoML](https://www.kdnuggets.com/2016/08/winning-automl-challenge-auto-sklearn.html). Pela simplicidade do propósito desta exploração, o escolhi para fazer alguns testes.

Algumas premissas importantes para interpretar melhor o que vem a seguir:
- Eu executei os _pipelines_ em meu computador pessoal e para fins de exploração. Sendo assim, não fiz longas e extensivas explorações dos recursos da biblioteca;
- De propósito, eu explorei a biblioteca de maneira mais _naive_ e tentando simular como pessoas sem tanta experiência neste tipo de modelagem fariam a exploração;

_Show me the code_.

```python
import autosklearn.classification
import yaml

with open('config.yml') as f:
    settings = yaml.load(f, Loader=yaml.FullLoader)

automl = autosklearn.classification.AutoSklearnClassifier(**settings)
automl.fit(train_features, train_target)
predicted_target = automl.predict(test_features)
print(automl.show_models())
print(automl.sprint_statistics())
print("Accuracy score", sklearn.metrics.accuracy_score(test_target, predicted_target))
```
Para executar os _pipelines_, eu montei as configurações em arquivos `yaml` com os parâmetros a serem usados pelo _pipeline_ do `automl`.

#### `config.yml` sem _ensemble_

Na primeira tentativa, vou tentar encontrar um modelo simples com apenas um conjunto de algoritmo + hiperparâmetros. Para isso, usei as configurações abaixo.

```yaml
n_jobs: 2
per_run_time_limit: 120 # 2 minutes
time_left_for_this_task: 1800 # 30 minutes 
include_preprocessors: ['no_preprocessing']
ensemble_size: 1 # to get only a model, not an ensemble
```

```
[(1.000000, SimpleClassificationPipeline({'balancing:strategy': 'none', 'classifier:__choice__': 'bernoulli_nb', 'data_preprocessing:categorical_transformer:categorical_encoding:__choice__': 'no_encoding', 'data_preprocessing:categorical_transformer:category_coalescence:__choice__': 'no_coalescense', 'data_preprocessing:numerical_transformer:imputation:strategy': 'mean', 'data_preprocessing:numerical_transformer:rescaling:__choice__': 'normalize', 'feature_preprocessor:__choice__': 'no_preprocessing', 'classifier:bernoulli_nb:alpha': 10.12857981579372, 'classifier:bernoulli_nb:fit_prior': 'False'},
dataset_properties={
  'task': 2,
  'sparse': False,
  'multilabel': False,
  'multiclass': True,
  'target_type': 'classification',
  'signed': False})),
]
```
O melhor modelo encontrado nesta execução foi o `bernoulli_nb` ([_Naive Bayes classifier for multivariate Bernoulli_](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html)). Até aí, nenhum problema.

As características da execução do _pipeline_ foram:
```
Metric: accuracy
Best validation score: 0.786458
Number of target algorithm runs: 1903
Number of successful target algorithm runs: 1880
Number of crashed target algorithm runs: 9
Number of target algorithms that exceeded the time limit: 5
Number of target algorithms that exceeded the memory limit: 9
```

Vamos às métricas:
```
Accuracy score: 0.7638709677419355
```

Ou seja, o modelo encontrado automaticamente pelo `auto-sklearn` foi **melhor** que o _vanilla Decision Tree_ porém **pior** que o _vanilla Random Forest_. 

#### `config.yml` com _ensemble_

Para explorar melhor as funcionalidades da biblioteca, criei um novo _pipeline_ onde busco como resultado final um [_ensemble_](https://en.wikipedia.org/wiki/Ensemble_learning) com 3 modelos para maximizar a acurácia da classificação. Abaixo seguem as configurações deste novo _pipeline_:

```yaml
n_jobs: 2
per_run_time_limit: 120 # 2 minutes
time_left_for_this_task: 1800 # 30 minutes 
include_preprocessors: ['no_preprocessing']
ensemble_size: 3 # to get only a model, not an ensemble
```

As características desse _ensemble_ foram:

```
[(0.333333, SimpleClassificationPipeline({'balancing:strategy': 'weighting', 'classifier:__choice__': 'sgd', 'data_preprocessing:categorical_transformer:categorical_encoding:__choice__': 'no_encoding', 'data_preprocessing:categorical_transformer:category_coalescence:__choice__': 'minority_coalescer', 'data_preprocessing:numerical_transformer:imputation:strategy': 'mean', 'data_preprocessing:numerical_transformer:rescaling:__choice__': 'quantile_transformer', 'feature_preprocessor:__choice__': 'no_preprocessing', 'classifier:sgd:alpha': 1.1309190654192295e-07, 'classifier:sgd:average': 'True', 'classifier:sgd:fit_intercept': 'True', 'classifier:sgd:learning_rate': 'optimal', 'classifier:sgd:loss': 'perceptron', 'classifier:sgd:penalty': 'elasticnet', 'classifier:sgd:tol': 0.0003060155962964433, 'data_preprocessing:categorical_transformer:category_coalescence:minority_coalescer:minimum_fraction': 0.006195858518768137, 'data_preprocessing:numerical_transformer:rescaling:quantile_transformer:n_quantiles': 1929, 'data_preprocessing:numerical_transformer:rescaling:quantile_transformer:output_distribution': 'uniform', 'classifier:sgd:l1_ratio': 0.0018872923177367703},
dataset_properties={
  'task': 2,
  'sparse': False,
  'multilabel': False,
  'multiclass': True,
  'target_type': 'classification',
  'signed': False})),
(0.333333, SimpleClassificationPipeline({'balancing:strategy': 'weighting', 'classifier:__choice__': 'extra_trees', 'data_preprocessing:categorical_transformer:categorical_encoding:__choice__': 'no_encoding', 'data_preprocessing:categorical_transformer:category_coalescence:__choice__': 'no_coalescense', 'data_preprocessing:numerical_transformer:imputation:strategy': 'mean', 'data_preprocessing:numerical_transformer:rescaling:__choice__': 'quantile_transformer', 'feature_preprocessor:__choice__': 'no_preprocessing', 'classifier:extra_trees:bootstrap': 'False', 'classifier:extra_trees:criterion': 'entropy', 'classifier:extra_trees:max_depth': 'None', 'classifier:extra_trees:max_features': 0.23993615625255216, 'classifier:extra_trees:max_leaf_nodes': 'None', 'classifier:extra_trees:min_impurity_decrease': 0.0, 'classifier:extra_trees:min_samples_leaf': 1, 'classifier:extra_trees:min_samples_split': 10, 'classifier:extra_trees:min_weight_fraction_leaf': 0.0, 'data_preprocessing:numerical_transformer:rescaling:quantile_transformer:n_quantiles': 1169, 'data_preprocessing:numerical_transformer:rescaling:quantile_transformer:output_distribution': 'uniform'},
dataset_properties={
  'task': 2,
  'sparse': False,
  'multilabel': False,
  'multiclass': True,
  'target_type': 'classification',
  'signed': False})),
(0.333333, SimpleClassificationPipeline({'balancing:strategy': 'weighting', 'classifier:__choice__': 'qda', 'data_preprocessing:categorical_transformer:categorical_encoding:__choice__': 'one_hot_encoding', 'data_preprocessing:categorical_transformer:category_coalescence:__choice__': 'no_coalescense', 'data_preprocessing:numerical_transformer:imputation:strategy': 'most_frequent', 'data_preprocessing:numerical_transformer:rescaling:__choice__': 'standardize', 'feature_preprocessor:__choice__': 'no_preprocessing', 'classifier:qda:reg_param': 0.40061768323123503},
dataset_properties={
  'task': 2,
  'sparse': False,
  'multilabel': False,
  'multiclass': True,
  'target_type': 'classification',
  'signed': False})),
]
```
Resumindo, o _ensemble_ é composto por um modelo [SGD - _Stochastic Gradient Descendent_](https://scikit-learn.org/stable/modules/sgd.html), um [_Extra Trees_](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html) e um [QDA - _Quadratic Discriminant Analysis_](https://scikit-learn.org/stable/modules/lda_qda.html), cada um com o respectivo conjunto de hiperparâmetros listados no _snippet_ acima. 

As características da execução do _pipeline_ foram:

```
Metric: accuracy
Best validation score: 0.786458
Number of target algorithm runs: 1949
Number of successful target algorithm runs: 1908
Number of crashed target algorithm runs: 10
Number of target algorithms that exceeded the time limit: 6
Number of target algorithms that exceeded the memory limit: 25
```

Vamos à acurácia:
```
Accuracy score: 0.7316129032258064
```

Ou seja, o _ensemble_ obteve uma acurácia de **73,2%** que, quando comparada aos casos anteriores, nos dá o seguinte _ranking_:
1. _Vanilla Random Forest_
2. AutoML sem _ensemble_
3. AutoML com _ensemble_
4. _Vanilla Decision Tree_

#### `config.yml` otimizando apenas o _Random Forest_

Como terceira tentativa, decidi explorar um _pipeline_ para otimização apenas do _Random Forest_ usando as configurações abaixo:

```yaml
n_jobs: 2
per_run_time_limit: 60 # 1 minute
time_left_for_this_task: 1200 # 20 minutes 
include_preprocessors: ['no_preprocessing']
include_estimators: ['random_forest'] # include only Random Forest
ensemble_size: 3 # to get only a model, not an ensemble
```

O melhor modelo encontrado pelo _pipeline_ foi:

```
[(1.000000, SimpleClassificationPipeline({'balancing:strategy': 'weighting', 'classifier:__choice__': 'random_forest', 'data_preprocessing:categorical_transformer:categorical_encoding:__choice__': 'no_encoding', 'data_preprocessing:categorical_transformer:category_coalescence:__choice__': 'minority_coalescer', 'data_preprocessing:numerical_transformer:imputation:strategy': 'mean', 'data_preprocessing:numerical_transformer:rescaling:__choice__': 'normalize', 'feature_preprocessor:__choice__': 'no_preprocessing', 'classifier:random_forest:bootstrap': 'False', 'classifier:random_forest:criterion': 'gini', 'classifier:random_forest:max_depth': 'None', 'classifier:random_forest:max_features': 0.14928991954179588, 'classifier:random_forest:max_leaf_nodes': 'None', 'classifier:random_forest:min_impurity_decrease': 0.0, 'classifier:random_forest:min_samples_leaf': 2, 'classifier:random_forest:min_samples_split': 8, 'classifier:random_forest:min_weight_fraction_leaf': 0.0, 'data_preprocessing:categorical_transformer:category_coalescence:minority_coalescer:minimum_fraction': 0.010000000000000004},
dataset_properties={
  'task': 2,
  'sparse': False,
  'multilabel': False,
  'multiclass': True,
  'target_type': 'classification',
  'signed': False})),
]
```

Ao olhar os parâmetros usados neste modelo, é possível observar que foram poucas as customizações de parâmetros realizadas, o que resulta em um modelo razoavelmente próximo ao _vanilla Random Forest_.

As características da execução do _pipeline_ foram:

```
Metric: accuracy
Best validation score: 0.786458
Number of target algorithm runs: 1260
Number of successful target algorithm runs: 1253
Number of crashed target algorithm runs: 0
Number of target algorithms that exceeded the time limit: 7
Number of target algorithms that exceeded the memory limit: 0
```

Os números acima mostram que o _pipeline_ executou mais de 1200 configurações diferentes de modelo, com alguns poucos casos que excederam o limite de tempo de 1 minuto por modelo.

Vamos às métricas:

```
Accuracy score: 0.8025806451612904
```

Ou seja, este _pipeline_ resultou em um modelo que alcançou uma acurácia de **80,3%**, o maior alcançado até o momento. Entretanto, o ganho não foi muito substancial (1,6%) e o resultado final foi um modelo muito parecido com o _vanilla_.

# Algumas conclusões e próximos passos

- Das opções testadas, o melhor modelo obtido foi a partir de um _pipeline_ do `auto-sklearn` otimizando apenas _Random Forests_. Entretanto, o resultado de acurácia é muito semelhante ao _vanilla Random Forest_;
- Há uma curva de aprendizado considerável para extrair o melhor da biblioteca;
- É possível que eu tenha dedicado pouco tempo de computação para este tipo de otimização pelo `auto-sklearn`. Se for o caso, acredito que este tipo de "necessidade" poderia ficar mais explícito nas documentações para que os usuários tivessem uma expectativa mais realista sobre o que precisariam para extrair valor da biblioteca;