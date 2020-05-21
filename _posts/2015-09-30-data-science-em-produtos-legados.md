---
layout: single
title:  "Usando Data Science em produtos legados"
author: Weslley
date:   2015-09-30
categories: data-science machine-learning

excerpt: "Neste artigo vamos falar um pouco sobre alguns pontos de atenção na integração de um produto que já existe há algum tempo e um modelo feito por _Data Scientists_."
header:
  overlay_color: "#367"
---

Sem dúvida alguma, _data science_ é um dos temas mais quentes do momento. Empresas dos mais diversos segmentos usam (ou querem usar) ciência de dados para melhorar seus produtos & processos, aumentando o valor percebido de seus clientes a um baixo custo operacional. A avalanche de novos produtos criados em plena era das startups já consideram os benefícios das informações extraídas dos sofisticados algoritmos. Entretanto, as empresas que já possuem produtos e operações há mais tempo tem um desafio a mais: como conectar meus produtos legados à tal inteligência?

Toda e qualquer alteração em um produto legado levanta a velha questão: qual o tamanho da alteração que devemos fazer? Mexer o mínimo possível? Reformular o produto por completo? Esta escolha não é simples e pode levar a dois cenários extremos: desperdício de esforço ou a reinvenção da roda.

O primeiro caminho, de alterar o mínimo possível, é o mais óbvio. Uma alteração mínima exige menos tempo e recurso, reduzindo o custo de desenvolvimento e o time-to-market. Entretanto, colocar o foco em questões de custo e integrações técnicas pode levar um time de data science a trabalhar por semanas a fio na otimização de uma funcionalidade que já estava obsoleta há anos, ou então na aplicação de regras de negócio legadas que invalidam boa parte da inteligência gerada pelos algoritmos. Além destes pontos, vale lembrar de uma ótima conclusão do [Data Jujitsu](https://www.oreilly.com/data/free/data-jujitsu.csp):

>"Resolver problemas de produto no front end é muito menos custoso que no back end." (DJ Patil)

Na outra extremidade, a reformulação completa de um produto é uma atitude mais arrojada, porém com riscos bastante altos, principalmente caso a base de clientes / usuários seja significativa. A base de clientes foi conquistada pela experiência de uso, visual, funcionalidades e valor entregue que o produto ofereceu. Uma mudança radical pode mudar demais um ou mais destes pontos e causar a perda de alguns clientes (_churn_). Além disso, uma mudança radical exige grandes esforços de desenvolvimento, aumentando custos e _time-to-market_.

Alguns pontos que aprendi e podem ajudar nesta tomada de decisão:

- Por mais que muitos sejam fascinados por temas como _machine learning_, redes neurais e inteligência artificial, o que garante o sucesso do produto são a conquista e a satisfação de clientes. Então mantenha o foco no cliente! Os dados são apenas uma ferramenta para melhorar o produto;

- O nível de especialidade e o salário dos profissionais de _data science_ são altos, enquanto a oferta de profissionais é pequena e concentrada em alguns polos. Portanto, falhas em desenvolvimentos de um time de data science custam caro para sua empresa;

- Alterações em _back-end_ são mais transparentes aos usuários do que em _front-end_. Dessa forma, gerar valor usando engines inteligentes pode aumentar o valor oferecido ao cliente com baixo impacto na utilização do produto propriamente dita.