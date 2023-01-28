# Projet semestriel : Hair segmentation project

## Auteurs : 
- Aymen Tilfani
- Badre Iddine Agtaib
- Nicolas Do

## Contenu

Ce dépôt contient le travail réalisé pour le projet semestriel traitant la segmentation de cheveux en temps réel. 

Il contient des scripts pour le traitement des datasets, le lancement des entraînement des modèles, les résultats et l'interface graphique.

## Utilisation

Les datasets utilisés sont Figaro1k, Lfw et une combinaison de ces deux derniers.

- Pour obtenir Figaro1k, Lfw et le dataset combiné:
Décompresser l'archive  `data.tgz` disponible [ici](https://filesender.renater.fr/?s=download&token=2b826262-7750-4c59-8260-de40755996d5)

Pour lancer l'entraînement de tous les modèles étudiés, lancer

- `train_all_model.sh`

Ces modèles sont disponible [ici](https://filesender.renater.fr/?s=download&token=bb5d1d00-ddf7-4332-ad80-5801c4fe70d6)

Sinon pour entraîner un modèle spécifique, lancer :

- `python3 run_train_segmentation.py [OPTIONS]`

|Option                         |   DEFAULT      |  Description                                                         |
|-------------------------------|----------------|----------------------------------------------------------------------|
|  --dataset DATASET            |     Figaro1k   |dataset used for training (Lfw, Figaro1k or Lfw+Figaro1k)             |
|  --test-dataset TEST_DATASET  |     all        |dataset used for testing (all, Lfw, Figaro1k or Lfw+Figaro1k)         |
|  --no-augmentation            |     False      |flag to disable data augmentation                                     |
|  --augmentation               |     True       |flag to enable data augmentation                                      |
|  --no-pretrained              |     True       |flag to disable use of pretrained encoder  (for mobile unet)          |
|  --pretrained                 |     False      |flag to enable use of pretrained encoder (for mobile unet)            |
|  --epochs EPOCHS              |      50        |Number of epochs to train the model for                               |
|  --size SIZE                  |      128       |Input size                                                            |
|  --model-type MODEL_TYPE      |     unet       |model used (unet or mobile_unet)                                      |


Pour évaluer tous les modèles entraînés, lancer :

- `python3 run_test_segmentation.py`. (Voir `python3 run_test_segmentation.py -h` pour évaluer un seul modèle en particulier)

Pour lancer l'interface graphique, lancer :

- `python3 app.py`



