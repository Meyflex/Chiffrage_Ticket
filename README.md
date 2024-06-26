﻿# Chiffrage_Ticket
Notre projet vise à prédire le chiffrage, en jours, des tickets sur Jira, une tâche complexe nécessitant à la fois une grande quantité de données qualitatives et un traitement minutieux de ces données pour en extraire des informations pertinentes. Nous avons commencé par évaluer la qualité de notre dataset, qui contenait environ 2000 entrées. Cette quantité limitée, ajoutée à la propreté discutable des données (incluant des incohérences et des informations superflues), représentait un défi significatif dès le début.

Pour aborder ce problème, nous avons opté pour une stratégie de régression initiale, expérimentant avec des modèles comme RandomForest, qui a montré des résultats prometteurs avec une précision de 7%. Cependant, même des modèles de machine learning plus avancés tels que BERT et DistilBERT, malgré leur puissance et leurs 340 millions de paramètres, n'ont pas surpassé RandomForest, atteignant des précisions de 3% et 2% respectivement. Cette situation a souligné l'importance cruciale de la quantité et de la qualité des données dans les problèmes de régression.

Face aux limites rencontrées avec la régression, nous avons décidé de simplifier le problème en adoptant une approche de classification entre trois classes : 0 à 3 jours, 3 à 8 jours, et plus de 8 jours. Cette simplification a significativement amélioré nos chances de succès. Avec un modèle de base comme Naive Bayes, nous avons obtenu des résultats encourageants autour de 45% de précision, qui ont légèrement augmenté à 50% après avoir oversamplé le dataset. Le undersampling, en revanche, a produit des résultats moins satisfaisants, avec une précision de 39%.

Dans notre quête d'améliorer davantage la performance, nous avons testé des modèles plus puissants tels que BERT et RoBERTa, mais leurs résultats ont été décevants, plafonnant à 40% et 43% de précision respectivement. Nous avons même tenté d'utiliser GPT-J, un modèle de 7 milliards de paramètres équivalent à ChatGPT-3, sans constater d'amélioration notable.

Le dénominateur commun de ces expériences a été la difficulté à obtenir une convergence adéquate de nos modèles, un problème que nous attribuons à la qualité et à la quantité insuffisantes de nos données. Malgré diverses tentatives de préparation et de nettoyage du dataset, nous sommes systématiquement arrivés à une précision plafonnée à environ 45%.

En conclusion, notre exploration a révélé que, malgré l'utilisation de modèles de machine learning de pointe et de stratégies de traitement des données innovantes, le succès de tels projets dépend intrinsèquement de la disponibilité de données abondantes et de haute qualité. Nos efforts pour prédire le chiffrage des tickets Jira ont mis en évidence les limites actuelles de notre dataset et la nécessité d'une collecte de données plus robuste pour soutenir des analyses prédictives précises.

Ce dépôt contient le code de ce projet, organisé en deux dossiers principaux : regression et classification. Chaque projet dispose de deux notebooks Jupyter pour une exploration interactive et pour le projet de classification, une classe main a été développée pour faciliter l'utilisation de modèles plus puissants, les notebooks n'étant pas suffisants pour exploiter leur plein potentiel.

Afin de garder la sécurité et la confidentialité du projet le dataset n’est pas ajoute sur le repos.

