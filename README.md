README - RAGV2 : Retrieval Augmented Generation
Ce notebook implémente un système complet de Retrieval Augmented Generation (RAG) permettant d’interroger des documents en utilisant un LLM (Large Language Model) avec un contexte pertinent extrait d’une base de données vectorielle.

Le projet illustre les étapes clés de la construction d’un pipeline RAG : indexation des documents, découpage en chunks, création d’embeddings, stockage vectoriel et requêtage avec injection de contexte dans un prompt.

Objectif
Fournir un exemple pratique et reproductible de RAG, en expliquant les concepts fondamentaux :

Indexation et vectorisation de documents

Choix d’un modèle d’embeddings (gte-large)

Stratégie de découpage (chunking) avec chevauchement

Recherche par similarité dans une base vectorielle (ChromaDB)

Génération augmentée par le contexte récupéré

Contenu du notebook
Le notebook suit le plan suivant :

Introduction au RAG
Présentation des limites des LLM et de l’intérêt du RAG.

Composants d’un système RAG

Indexation

Retrieval (récupération)

Génération

Étape 1 : Création d’une base de données vectorielle (Indexation)

Choix d’un modèle d’embeddings (gte-large)

Découpage en chunks (taille fixe + chevauchement)

Stockage des chunks et de leurs embeddings dans ChromaDB

Étape 2 : Interrogation de la base vectorielle

Embedding de la requête

Recherche des chunks les plus similaires

Construction du prompt avec contexte et requête

Appel au LLM pour générer la réponse

Prérequis
Python 3.8+ recommandé

Connaissances de base en traitement du langage naturel et en LLM

Bibliothèques Python utilisées (à installer) :

langchain : pour le découpage, les retrievers et l’interface avec ChromaDB

chromadb : base de données vectorielle

sentence-transformers : pour le modèle d’embeddings gte-large

pypdf (ou autre) pour le parsing de PDF (selon les documents)

transformers, torch (pour le LLM, si utilisé localement)

Installation
bash
git clone <url-du-dépôt>
cd <dossier-projet>
pip install -r requirements.txt
Exemple de requirements.txt (indicatif) :

text
langchain
chromadb
sentence-transformers
pypdf
transformers
torch
Utilisation
Placez vos documents (PDF, texte, etc.) dans le répertoire indiqué.

Ouvrez le notebook RAGV2.ipynb.

Exécutez les cellules dans l’ordre pour :

Charger les documents

Découper en chunks

Créer les embeddings et la base vectorielle

Poser une question et obtenir une réponse augmentée

Note : Vous pouvez modifier le modèle d’embeddings, la taille des chunks, le chevauchement, ou le LLM utilisé (via l’API d’OpenAI ou localement).

Fonctionnalités
Parsing de documents (PDF supporté)

Chunking paramétrable (taille, chevauchement)

Embeddings via gte-large (ou tout autre modèle de HuggingFace)

Base vectorielle persistante avec ChromaDB

Recherche par similarité cosinus

Génération avec injection dynamique du contexte
