---
layout: single
title:  "_Statistical Learning Theory_ — você deveria usar mais do que imagina"
author: Weslley
date:   2017-07-25
categories: data-science machine-learning

excerpt: "Neste artigo vamos falar um pouco sobre alguns conceitos básicos da Teoria do Aprendizado Estatístico e como isso é importante no dia-a-dia de um _Data Scientist_."
header:
  overlay_image: /assets/images/matrix_background.jpg
  overlay_filter: 0.7
---

Tem aumentado a frequência em que eu escuto profissionais e simpatizantes de _Data Science_ minimizando (ou até desprezando) a importância de conhecimento em Estatística para a criações de soluções baseadas em Inteligência Artificial e/ou _Machine Learning_. Isso sempre me remete a mesma preocupação com relação a uma etapa importante da modelagem e que está cada vez mais sendo negligenciada: a validação de algumas premissas necessárias para que ocorra aprendizado supervisionado em um domínio de dados.

Por conta disso, eu gostaria de trazer à tona alguns conceitos básicos de _Statistical Learning Theory_ que abordam algumas premissas que um bom conjunto de dados pode ter para que uma seja possível a construção preditiva baseada em dados, ou seja, para garantir que um algoritmo será capaz de aprender corretamente.

![](/assets/images/tirinha-bias.gif){: .align-center}
*http://letterstoayounglibrarian.blogspot.com/2016/04/on-bias.html*
{: .text-center}


Usando uma linguagem mais cotidiana, estamos falando do famoso _BIAS_, que ocorre quando uma de suas variáveis contém a distribuição de probabilidade de seu alvo. Por exemplo, se você está tentando prever se uma pessoa é completamente careca ou não, ter uma variável que contém o número de fios de cabelo que cada indivíduo tem incluirá a distribuição de carecas no treinamento, dado que o fator que determina a classe do indivíduo é ele ter fios de cabelo ou não. Portanto, seu algoritmo lhe fornecerá um diagnóstico, e não uma previsão.

Para mitigar tal tipo de problema, uma análise de covariância ou de dependência linear entre suas variáveis e seu alvo podem lhe ajudar a identificar a ocorrência de _BIAS_.

## A distribuição de probabilidade de seu alvo deve ser estática

![](/assets/images/static-tv.jpg){: .align-center}
*https://www.slashgear.com/one-percent-of-tv-static-originates-from-the-big-bang-24492754/*
{: .text-center}

Em outras palavras, quero dizer que seus dados não podem ter dependência temporal dentro das condições de contorno de seu problema. Caso não seja, seu algoritmo aprenderá a construir uma função de predição que não será mais a mesma após determinado intervalo de tempo, e haverá degradação de seu poder de predição.

Para casos de variação temporal dentro de suas condições de contorno, abordagens como Séries Temporais são mais adequadas.

## As amostras usadas para aprendizado devem representar todo o seu espaço amostral

![](/assets/images/diversity.jpg){: .align-center}
*https://www.theodysseyonline.com/5-reasons-why-representation-matters*
{: .text-center}

Ao escolher um conjunto de elementos para construir uma amostra de seu domínio de dados, você deve garantir que todo o seu espaço está representado por tais amostras, ou seja, de que existem amostras de todas as regiões de seu espaço e que a proporcionalidade dos grupos contidos neles também é mantida.

Imagine que você está na fila de um banco com outras 100 pessoas (que baita fila) e queira criar um grupo de indivíduos que represente todas as pessoas que ali estão. No final de sua amostragem, temos 10 indivíduos: 2 crianças de até 10 anos, 5 mulheres entre 25 e 35 anos, 3 homens entre 18 e 25 anos. Esta amostra lhe diria que:

- 20% das pessoas na fila são crianças;
- Não existem idosos na fila do banco;
- Há quase o dobro de mulheres na fila se comparadas aos homens;

Entretanto, se você olhar o espaço como um todo pode perceber que o espaço completo não está sendo bem representado, pois existiam idosos na fila, e as 2 crianças na amostra na verdade eram as duas únicas crianças em toda a fila. Este tipo de situação pode acontecer por três motivos principais:

- Uma amostragem feita de maneira não aleatória;
- Uma amostragem aleatória, porém com uma quantidade pequena o suficiente para não considerar alguns grupos minoritários contidos no espaço;
- Diferença entre as condições de contorno (ou segmentação) que definem o espaço completo e o amostral;

Para evitar tal tipo de situação, analisar a variância de sua amostra e compara-la com a mesma métrica do espaço completo pode lhe ajudar a evitar tal tipo de situação.

## As amostras devem ser independentes entre si

![](/assets/images/porta-esperanca.png){: .align-center}
*https://brilliant.org/practice/conditional-probability-in-quant-finance/*
{: .text-center}

Ao desenhar sua amostragem, os indivíduos de seu espaço amostral não podem ter dependência entre si, ou seja, a probabilidade de um evento acontecer para uma indivíduo não pode ser condicional a outro indivíduo ou então exigir que seu espaço amostral contenha determinados indivíduos.

Um bom exemplo é a boa e velha predição de churn. Suponha que você trabalha com um produto cujo modelo de negócio é uma assinatura mensal que permite uma estrutura de titular e dependentes do plano. Ao tentar modelar sua predição tendo como alvo o churn de um usuário, automaticamente você terá dependência condicional em seu espaço, já que a assinatura, alterações contratuais e cancelamentos ocorrerão sobre todos os usuários que estão contidos em uma conta.

Em termos mais técnicos, tal condicionamento é ruim para o aprendizado pois a sua distribuição de probabilidade do alvo dependerá de escolhas específicas de sua amostragem. Para evitar tal tipo de situação, você deve identificar muito bem qual é a melhor representação para o indivíduo de seu espaço (que no exemplo acima seria uma conta, e não um usuário).

## Seu alvo pode ter ruído

![](/assets/images/the-germs.jpg)
*The Germs: ruído de qualidade - http://www.latimes.com/entertainment/arts/miranda/la-et-cam-slash-magazine-book-exhibition-20160719-snap-story.html*
{: .text-center}

Por último, mas não menos importante, vem o fato de que seu alvo, representado pela classe/valor alvo de cada indivíduo em seu espaço amostral, pode ter ruído, ou seja, erros de classificação no ground truth, desde que sejam minoritários.

Este item é importante pois é muito comum que alguns profissionais, durante a validação da qualidade dos dados e variáveis, encontre certo ruído e sintam-se tentados a remover os indivíduos ruidoso de seu espaço amostral. Ao fazer isso, você estará alterando as características de seu espaço e impactando as capacidades de seu modelo. Muito provavelmente você terá que processar novas entradas que possuem este perfil problemático (por exemplo, erros em sistemas de cadastro feitos manualmente) sem ter permitido que seu modelo tivesse amostras desse perfil em seu conjunto de dados de treinamento. Lembrando que a premissa é que tal ruído seja minoritário e será tratado como tal pela predição.

Repare que todos os pontos abordados acima estão diretamente relacionados a conceitos estatísticos (e a maior parte deles não muito avançados). Sendo assim, se você não enxerga a importância de tais conceitos em seu dia-a-dia, pode ser que algumas etapas importantes de seu trabalho estejam sendo negligenciadas.
