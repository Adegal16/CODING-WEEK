# Rapport de Contribution - Explicabilité et Évaluation (Membre 3)

## 1. Ma contribution principale : Implémentation de SHAP
Mon rôle principal a été de rendre notre modèle de prédiction du risque d'insuffisance cardiaque transparent et interprétable pour les médecins. J'ai développé entièrement le code source dans `src/evaluate_model.py` :

**Analyse Globale (Summary Plot) :** J'ai codé la fonction `generate_shap_summary` qui permet d'identifier quelles sont les caractéristiques médicales les plus influentes sur les prédictions du modèle à l'échelle de toute la base de patients.
***Analyse Individuelle (Waterfall Plot) :** J'ai développé la fonction `generate_shap_individual` pour expliquer le diagnostic d'un patient précis. J'ai spécifiquement adapté l'extraction des valeurs SHAP (`shap_values_patient[0, :, 1]`) pour isoler correctement la probabilité de la classe positive (le risque effectif de maladie), gérant ainsi la sortie matricielle des modèles de classification.
***Intégration Interface :** J'ai structuré ces fonctions pour qu'elles retournent directement des objets figures (`plt.gcf()`), facilitant ainsi leur intégration par le Membre 4 dans l'interface Streamlit[cite: 52, 80].

## 2. Évaluation des performances du Modèle
J'ai également mis en place le pipeline d'évaluation des modèles entraînés.J'ai créé la fonction `evaluate_metrics` qui calcule et retourne les indicateurs clés de performance exigés par le projet : l'exactitude (Accuracy), la précision, le rappel, le F1-score, et surtout le ROC-AUC.

---

## 3. Ingénierie des Prompts (Test Automatisé)
Conformément aux exigences du projet, j'ai utilisé l'assistance de l'IA (LLM) **exclusivement** pour une tâche spécifique : la création des tests automatisés avec `pytest`.

***Contexte :** Ayant rédigé le code SHAP, j'avais besoin de vérifier sa robustesse avec des données fictives avant de l'intégrer au workflow CI/CD.
***Résultats obtenus :** L'IA m'a fourni la structure du fichier `test_evaluate_model.py`, m'aidant à comprendre comment simuler un DataFrame de test et un modèle `RandomForestClassifier` basique pour vérifier que mes fonctions retournaient bien des objets graphiques valides (`plt.Figure`).
***Résolution de bugs assistée :** J'ai également soumis à l'IA l'erreur `ValueError: The waterfall plot can currently only plot a single explanation` obtenue lors du premier passage du test, ce qui m'a aidé à diagnostiquer le conflit de dimensionnalité entre la sortie multiclasse du modèle et l'attente du graphique Waterfall.