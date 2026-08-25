# Atelier Modele De Scoring

Ce document sert de support de travail pour reconstruire le modele de scoring Next Legend avec des coachs, scouts et professionnels de l'industrie.

Objectif : remplacer le scoring par roles, par exemple `Ball-Playing Centre Back`, par un scoring par groupes de postes, par exemple `Centre Backs`, `Left Backs` ou `Centre Forwards`.

Apres l'atelier, complete directement ce fichier et renvoie-le a Codex. Codex transformera le modele valide en configuration exploitable par le projet et adaptera le pipeline.

## Comment Remplir Ce Fichier

Pour chaque groupe de postes :
1. Garder, supprimer ou ajouter des metriques depuis la liste proposee.
2. Definir un poids pour chaque metrique.
3. Marquer les metriques ou une valeur plus basse est meilleure.
4. Garder le total des poids metriques a `100`.
5. Ajouter des commentaires lorsqu'une decision est tactique ou subjective.

Utiliser exactement la cle technique de metrique ecrite entre backticks. Le libelle sert uniquement a la lecture humaine.

Exemple :

| Cle metrique | Libelle | Sens | Poids | Garder ? | Notes |
| --- | --- | --- | ---: | --- | --- |
| `def_duels_won_percent` | Duels defensifs gagnes (%) | plus haut | 20 | oui | Metrique centrale de qualite dans les duels |

## Entrees Globales Du Scoring

Le nouveau modele doit inclure quatre blocs de scoring :

| Bloc | Proposition de depart | Valeur atelier | Notes |
| --- | ---: | ---: | --- |
| Metriques du poste | 75 |  | Metriques techniques/tactiques ponderees pour le groupe de postes |
| Fiabilite des minutes | 10 |  | Penalise les tres petits echantillons, surtout en fin de saison |
| Coefficient de force de la ligue | 10 |  | Valorise la performance dans des competitions plus fortes |
| Coefficient de force de l'equipe | 5 |  | Ajuste le contexte equipe |
| Total | 100 | 100 | Doit totaliser 100 |

Logique de depart recommandee :
- Les metriques du poste restent le coeur du modele.
- Les minutes doivent etre un ajustement de fiabilite, pas une pure metrique de performance.
- Le coefficient ligue doit compter davantage que le coefficient equipe.
- Le coefficient equipe doit eviter de sur-noter les joueurs d'equipes dominantes et de sous-noter les joueurs d'equipes faibles ; le sens exact doit etre defini pendant l'atelier.

## Fiabilite Des Minutes

Decision a prendre :

| Parametre | Proposition | Valeur atelier | Notes |
| --- | ---: | ---: | --- |
| Minutes minimum visibles | 90 |  | Le joueur peut apparaitre mais doit etre fortement penalise |
| Echantillon stable | 900 |  | Environ 10 matchs complets |
| Minutes de pleine confiance | 1800 |  | Environ 20 matchs complets |
| Reference de fin de saison | 3000 |  | Sert a identifier les temps de jeu tres faibles sur une saison complete |
| Bonus maximum lie aux minutes | 0 |  | Recommande : pas de bonus, seulement une penalite de fiabilite |

Interpretation recommandee :
- `0-90` minutes : preuve insuffisante.
- `90-600` minutes : forte penalite de fiabilite.
- `600-1200` minutes : penalite moderee.
- `1200-1800` minutes : penalite legere.
- `1800+` minutes : pleine confiance.

Questions ouvertes pour l'atelier :
- Faut-il recompenser la disponibilite/durabilite, ou les minutes doivent-elles seulement reduire la confiance sur les petits echantillons ?
- Les gardiens doivent-ils avoir des seuils differents ?
- Les jeunes joueurs doivent-ils avoir des seuils de minutes plus souples ?

## Coefficients Ligue Et Equipe

La force de la ligue existe deja dans la logique de scoring. L'atelier doit decider a quel point elle influence le score final.

La force de l'equipe doit etre clairement interpretee footballistiquement avant implementation :

| Option | Signification | Risque |
| --- | --- | --- |
| Recompense du contexte equipe forte | Un joueur performant dans une grosse equipe recoit un boost | Peut sur-noter les joueurs d'equipes dominantes |
| Recompense du contexte equipe faible | Un joueur performant dans une equipe faible recoit un boost contextuel | Peut sur-noter les joueurs a gros volume defensif |
| Correction contextuelle neutre | Le coefficient equipe ajuste seulement les contextes extremes | Plus complexe, mais generalement plus juste |

Decision atelier :

| Entree | Sens | Poids | Notes |
| --- | --- | ---: | --- |
| Coefficient de force de la ligue | coefficient plus haut = score plus haut |  |  |
| Coefficient de force de l'equipe | a definir |  |  |

## Groupes De Postes

Groupes cibles :

| Cle groupe | Nom affiche | Postes inclus |
| --- | --- | --- |
| `goalkeepers` | Gardiens | GK |
| `centre_backs` | Defenseurs centraux | CB, LCB, RCB |
| `left_backs` | Lateraux gauches | LB, LWB |
| `right_backs` | Lateraux droits | RB, RWB |
| `defensive_midfielders` | Milieux defensifs | DMF, LDMF, RDMF |
| `central_midfielders` | Milieux centraux | LCMF, RCMF |
| `attacking_midfielders` | Milieux offensifs | AMF, LAMF, RAMF |
| `left_wingers` | Ailiers gauches | LW, LWF |
| `right_wingers` | Ailiers droits | RW, RWF |
| `centre_forwards` | Avant-centres | CF |

Ajouter ou modifier des groupes ici :

| Cle groupe | Nom affiche | Postes inclus | Notes |
| --- | --- | --- | --- |
|  |  |  |  |

## Sens Des Metriques

La plupart des metriques sont meilleures lorsqu'elles sont plus hautes. Voici les candidates courantes ou une valeur plus basse est meilleure :

| Cle metrique | Libelle | Usage recommande |
| --- | --- | --- |
| `goals_conceded_per_90` | Buts encaisses /90 | Gardiens uniquement, tres contextuel |
| `xg_against_per_90` | xGA /90 | Gardiens uniquement, tres contextuel |
| `fouls_per_90` | Fautes /90 | Discipline defensive |
| `yellow_cards_per_90` | Cartons jaunes /90 | Discipline si disponible |
| `red_cards_per_90` | Cartons rouges /90 | Discipline si disponible |

## Gardiens

Postes inclus : GK

Idee de modele : equilibrer arrets, prevention, relance et maitrise de la surface.

| Cle metrique | Libelle | Sens | Poids propose | Poids atelier | Garder ? | Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| `save_percent` | Arrets (%) | plus haut | 20 |  |  | Qualite d'arret |
| `goals_prevented_per_90` | Buts evites /90 | plus haut | 18 |  |  | Arrets au-dessus de l'attendu |
| `goals_conceded_per_90` | Buts encaisses /90 | plus bas | 8 |  |  | Tres contextuel |
| `xg_against_per_90` | xGA /90 | plus bas | 6 |  |  | Contexte defensif de l'equipe |
| `exits_per_90` | Sorties /90 | plus haut | 10 |  |  | Activite de sweeper / surface |
| `aerial_duels_gk_per_90` | Duels aeriens GK /90 | plus haut | 8 |  |  | Maitrise de la surface |
| `passes_per_90` | Passes /90 | plus haut | 8 |  |  | Implication dans la relance |
| `accurate_passes_percent` | Precision de passe (%) | plus haut | 8 |  |  | Securite courte/moyenne |
| `long_passes_per_90` | Passes longues /90 | plus haut | 7 |  |  | Portee de relance |
| `accurate_long_passes_percent` | Precision passes longues (%) | plus haut | 7 |  |  | Qualite de relance longue |

Metriques additionnelles a considerer :
- `clean_sheets`
- `shots_against_per_90`
- `back_passes_to_gk_per_90`
- `avg_pass_length_m`

## Defenseurs Centraux

Postes inclus : CB, LCB, RCB

Idee de modele : securite dans les duels, domination aerienne, activite defensive, anticipation et progression depuis l'arriere.

| Cle metrique | Libelle | Sens | Poids propose | Poids atelier | Garder ? | Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| `def_duels_won_percent` | Duels defensifs gagnes (%) | plus haut | 16 |  |  | Qualite defensive centrale |
| `def_duels_per_90` | Duels defensifs /90 | plus haut | 8 |  |  | Volume d'engagement defensif |
| `successful_def_actions_per_90` | Actions defensives reussies /90 | plus haut | 12 |  |  | Activite defensive globale |
| `interceptions_per_90` | Interceptions /90 | plus haut | 10 |  |  | Lecture / anticipation |
| `aerial_duels_won_percent` | Duels aeriens gagnes (%) | plus haut | 14 |  |  | Qualite aerienne |
| `aerial_duels_per_90` | Duels aeriens /90 | plus haut | 8 |  |  | Volume aerien |
| `progressive_passes_per_90` | Passes progressives /90 | plus haut | 10 |  |  | Progression par la passe |
| `accurate_progressive_passes_percent` | Precision passes progressives (%) | plus haut | 6 |  |  | Qualite de progression |
| `long_passes_per_90` | Passes longues /90 | plus haut | 6 |  |  | Jeu long |
| `accurate_long_passes_percent` | Precision passes longues (%) | plus haut | 5 |  |  | Qualite du jeu long |
| `progressive_runs_per_90` | Courses progressives /90 | plus haut | 5 |  |  | Conduite vers l'avant |

Metriques additionnelles a considerer :
- `sliding_tackles_per_90`
- `passes_per_90`
- `accurate_passes_percent`
- `passes_to_final_third_per_90`
- `fouls_per_90`

## Lateraux Gauches

Postes inclus : LB, LWB

Idee de modele : progression laterale, centres, contribution dans le dernier tiers et fiabilite defensive en 1v1.

| Cle metrique | Libelle | Sens | Poids propose | Poids atelier | Garder ? | Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| `def_duels_won_percent` | Duels defensifs gagnes (%) | plus haut | 12 |  |  | Defense en 1v1 |
| `successful_def_actions_per_90` | Actions defensives reussies /90 | plus haut | 10 |  |  | Activite defensive |
| `interceptions_per_90` | Interceptions /90 | plus haut | 8 |  |  | Lecture |
| `progressive_runs_per_90` | Courses progressives /90 | plus haut | 12 |  |  | Progression balle au pied |
| `progressive_passes_per_90` | Passes progressives /90 | plus haut | 8 |  |  | Progression par la passe |
| `passes_to_final_third_per_90` | Passes vers dernier tiers /90 | plus haut | 8 |  |  | Gain territorial |
| `crosses_per_90` | Centres /90 | plus haut | 10 |  |  | Volume de creation laterale |
| `accurate_crosses_percent` | Precision des centres (%) | plus haut | 8 |  |  | Qualite de creation laterale |
| `deep_crosses_per_90` | Centres profonds /90 | plus haut | 6 |  |  | Service avance depuis le cote |
| `dribbles_per_90` | Dribbles /90 | plus haut | 8 |  |  | 1v1 offensif |
| `xa_per_90` | xA /90 | plus haut | 10 |  |  | Qualite des occasions creees |

Metriques additionnelles a considerer :
- `passes_to_penalty_area_per_90`
- `successful_dribbles_percent`
- `accelerations_per_90`
- `fouls_per_90`
- `aerial_duels_won_percent`

## Lateraux Droits

Postes inclus : RB, RWB

Idee de modele : meme modele que les lateraux gauches, sauf si le club souhaite un scoring asymetrique.

| Cle metrique | Libelle | Sens | Poids propose | Poids atelier | Garder ? | Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| `def_duels_won_percent` | Duels defensifs gagnes (%) | plus haut | 12 |  |  | Defense en 1v1 |
| `successful_def_actions_per_90` | Actions defensives reussies /90 | plus haut | 10 |  |  | Activite defensive |
| `interceptions_per_90` | Interceptions /90 | plus haut | 8 |  |  | Lecture |
| `progressive_runs_per_90` | Courses progressives /90 | plus haut | 12 |  |  | Progression balle au pied |
| `progressive_passes_per_90` | Passes progressives /90 | plus haut | 8 |  |  | Progression par la passe |
| `passes_to_final_third_per_90` | Passes vers dernier tiers /90 | plus haut | 8 |  |  | Gain territorial |
| `crosses_per_90` | Centres /90 | plus haut | 10 |  |  | Volume de creation laterale |
| `accurate_crosses_percent` | Precision des centres (%) | plus haut | 8 |  |  | Qualite de creation laterale |
| `deep_crosses_per_90` | Centres profonds /90 | plus haut | 6 |  |  | Service avance depuis le cote |
| `dribbles_per_90` | Dribbles /90 | plus haut | 8 |  |  | 1v1 offensif |
| `xa_per_90` | xA /90 | plus haut | 10 |  |  | Qualite des occasions creees |

Metriques additionnelles a considerer :
- `passes_to_penalty_area_per_90`
- `successful_dribbles_percent`
- `accelerations_per_90`
- `fouls_per_90`
- `aerial_duels_won_percent`

## Milieux Defensifs

Postes inclus : DMF, LDMF, RDMF

Idee de modele : recuperation, anticipation, securite dans les duels et distribution sure/progressive.

| Cle metrique | Libelle | Sens | Poids propose | Poids atelier | Garder ? | Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| `successful_def_actions_per_90` | Actions defensives reussies /90 | plus haut | 16 |  |  | Activite defensive |
| `interceptions_per_90` | Interceptions /90 | plus haut | 14 |  |  | Lecture et placement |
| `def_duels_won_percent` | Duels defensifs gagnes (%) | plus haut | 12 |  |  | Qualite dans les duels |
| `def_duels_per_90` | Duels defensifs /90 | plus haut | 8 |  |  | Volume d'engagement |
| `aerial_duels_won_percent` | Duels aeriens gagnes (%) | plus haut | 6 |  |  | Securite physique |
| `passes_per_90` | Passes /90 | plus haut | 8 |  |  | Implication |
| `accurate_passes_percent` | Precision de passe (%) | plus haut | 8 |  |  | Securite |
| `progressive_passes_per_90` | Passes progressives /90 | plus haut | 12 |  |  | Valeur vers l'avant |
| `accurate_progressive_passes_percent` | Precision passes progressives (%) | plus haut | 8 |  |  | Qualite vers l'avant |
| `passes_to_final_third_per_90` | Passes vers dernier tiers /90 | plus haut | 8 |  |  | Casser des lignes |

Metriques additionnelles a considerer :
- `sliding_tackles_per_90`
- `forward_passes_per_90`
- `accurate_forward_passes_percent`
- `long_passes_per_90`
- `fouls_per_90`

## Milieux Centraux

Postes inclus : LCMF, RCMF

Idee de modele : progression, implication dans les deux phases, portee balle au pied, lien avec le dernier tiers et contribution aux buts.

| Cle metrique | Libelle | Sens | Poids propose | Poids atelier | Garder ? | Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| `progressive_passes_per_90` | Passes progressives /90 | plus haut | 12 |  |  | Progression par la passe |
| `passes_to_final_third_per_90` | Passes vers dernier tiers /90 | plus haut | 10 |  |  | Gain territorial |
| `progressive_runs_per_90` | Courses progressives /90 | plus haut | 10 |  |  | Progression balle au pied |
| `dribbles_per_90` | Dribbles /90 | plus haut | 6 |  |  | Conduite / 1v1 |
| `successful_dribbles_percent` | Dribbles reussis (%) | plus haut | 5 |  |  | Qualite de dribble |
| `successful_def_actions_per_90` | Actions defensives reussies /90 | plus haut | 10 |  |  | Travail defensif |
| `def_duels_won_percent` | Duels defensifs gagnes (%) | plus haut | 8 |  |  | Qualite dans les duels |
| `key_passes_per_90` | Passes cles /90 | plus haut | 8 |  |  | Creation d'occasions |
| `xa_per_90` | xA /90 | plus haut | 8 |  |  | Qualite des occasions creees |
| `goals_per_90` | Buts /90 | plus haut | 8 |  |  | Presence dans la surface |
| `assists_per_90` | Passes decisives /90 | plus haut | 7 |  |  | Production finale |
| `accurate_passes_percent` | Precision de passe (%) | plus haut | 8 |  |  | Securite balle |

Metriques additionnelles a considerer :
- `accelerations_per_90`
- `offensive_duels_per_90`
- `smart_passes_per_90`
- `passes_to_penalty_area_per_90`

## Milieux Offensifs

Postes inclus : AMF, LAMF, RAMF

Idee de modele : creation d'occasions, passes dans le dernier tiers, menace dans la surface et jeu de combinaison.

| Cle metrique | Libelle | Sens | Poids propose | Poids atelier | Garder ? | Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| `xa_per_90` | xA /90 | plus haut | 16 |  |  | Qualite des occasions creees |
| `key_passes_per_90` | Passes cles /90 | plus haut | 14 |  |  | Creation d'occasions |
| `smart_passes_per_90` | Smart passes /90 | plus haut | 10 |  |  | Passes a haute valeur |
| `through_passes_per_90` | Passes en profondeur /90 | plus haut | 8 |  |  | Penetration |
| `deep_completions_per_90` | Deep completions /90 | plus haut | 8 |  |  | Jeu avance |
| `passes_to_penalty_area_per_90` | Passes vers surface /90 | plus haut | 8 |  |  | Acces a la surface |
| `progressive_passes_per_90` | Passes progressives /90 | plus haut | 8 |  |  | Progression |
| `touches_in_penalty_area_per_90` | Touches dans la surface /90 | plus haut | 8 |  |  | Presence dans la surface |
| `xg_per_90` | xG /90 | plus haut | 8 |  |  | Menace au tir |
| `goals_per_90` | Buts /90 | plus haut | 6 |  |  | Production |
| `dribbles_per_90` | Dribbles /90 | plus haut | 6 |  |  | 1v1 / portee |

Metriques additionnelles a considerer :
- `shot_assists_per_90`
- `progressive_runs_per_90`
- `assists_per_90`
- `successful_dribbles_percent`

## Ailiers Gauches

Postes inclus : LW, LWF

Idee de modele : menace en 1v1, progression, creation d'occasions, menace dans la surface et contribution defensive si souhaitee.

| Cle metrique | Libelle | Sens | Poids propose | Poids atelier | Garder ? | Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| `dribbles_per_90` | Dribbles /90 | plus haut | 14 |  |  | Volume de 1v1 |
| `successful_dribbles_percent` | Dribbles reussis (%) | plus haut | 8 |  |  | Qualite en 1v1 |
| `progressive_runs_per_90` | Courses progressives /90 | plus haut | 12 |  |  | Progression balle au pied |
| `accelerations_per_90` | Accelerations /90 | plus haut | 6 |  |  | Dynamisme |
| `crosses_per_90` | Centres /90 | plus haut | 8 |  |  | Creation laterale |
| `accurate_crosses_percent` | Precision des centres (%) | plus haut | 6 |  |  | Qualite laterale |
| `xa_per_90` | xA /90 | plus haut | 10 |  |  | Qualite des occasions creees |
| `key_passes_per_90` | Passes cles /90 | plus haut | 8 |  |  | Creation d'occasions |
| `touches_in_penalty_area_per_90` | Touches dans la surface /90 | plus haut | 8 |  |  | Menace surface |
| `xg_per_90` | xG /90 | plus haut | 8 |  |  | Qualite des tirs |
| `goals_per_90` | Buts /90 | plus haut | 8 |  |  | Production |
| `successful_def_actions_per_90` | Actions defensives reussies /90 | plus haut | 4 |  |  | Travail defensif optionnel |

Metriques additionnelles a considerer :
- `offensive_duels_per_90`
- `deep_crosses_per_90`
- `shots_per_90`
- `shots_on_target_percent`

## Ailiers Droits

Postes inclus : RW, RWF

Idee de modele : meme modele que les ailiers gauches, sauf si le club souhaite des attentes specifiques par cote.

| Cle metrique | Libelle | Sens | Poids propose | Poids atelier | Garder ? | Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| `dribbles_per_90` | Dribbles /90 | plus haut | 14 |  |  | Volume de 1v1 |
| `successful_dribbles_percent` | Dribbles reussis (%) | plus haut | 8 |  |  | Qualite en 1v1 |
| `progressive_runs_per_90` | Courses progressives /90 | plus haut | 12 |  |  | Progression balle au pied |
| `accelerations_per_90` | Accelerations /90 | plus haut | 6 |  |  | Dynamisme |
| `crosses_per_90` | Centres /90 | plus haut | 8 |  |  | Creation laterale |
| `accurate_crosses_percent` | Precision des centres (%) | plus haut | 6 |  |  | Qualite laterale |
| `xa_per_90` | xA /90 | plus haut | 10 |  |  | Qualite des occasions creees |
| `key_passes_per_90` | Passes cles /90 | plus haut | 8 |  |  | Creation d'occasions |
| `touches_in_penalty_area_per_90` | Touches dans la surface /90 | plus haut | 8 |  |  | Menace surface |
| `xg_per_90` | xG /90 | plus haut | 8 |  |  | Qualite des tirs |
| `goals_per_90` | Buts /90 | plus haut | 8 |  |  | Production |
| `successful_def_actions_per_90` | Actions defensives reussies /90 | plus haut | 4 |  |  | Travail defensif optionnel |

Metriques additionnelles a considerer :
- `offensive_duels_per_90`
- `deep_crosses_per_90`
- `shots_per_90`
- `shots_on_target_percent`

## Avant-Centres

Postes inclus : CF

Idee de modele : presence dans la surface, qualite des occasions, finition, contribution aerienne / point d'appui et jeu de lien.

| Cle metrique | Libelle | Sens | Poids propose | Poids atelier | Garder ? | Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| `xg_per_90` | xG /90 | plus haut | 16 |  |  | Qualite des occasions |
| `goals_per_90` | Buts /90 | plus haut | 16 |  |  | Production |
| `shots_per_90` | Tirs /90 | plus haut | 8 |  |  | Volume de tirs |
| `shots_on_target_percent` | Tirs cadres (%) | plus haut | 6 |  |  | Qualite des tirs |
| `goal_conversion_rate` | Taux de conversion (%) | plus haut | 6 |  |  | Efficacite de finition |
| `touches_in_penalty_area_per_90` | Touches dans la surface /90 | plus haut | 12 |  |  | Presence surface |
| `aerial_duels_won_percent` | Duels aeriens gagnes (%) | plus haut | 8 |  |  | Valeur de point d'appui |
| `aerial_duels_per_90` | Duels aeriens /90 | plus haut | 6 |  |  | Volume point d'appui |
| `passes_received_per_90` | Passes recues /90 | plus haut | 6 |  |  | Implication |
| `long_passes_received_per_90` | Longues passes recues /90 | plus haut | 5 |  |  | Solution en jeu direct |
| `xa_per_90` | xA /90 | plus haut | 6 |  |  | Jeu de lien |
| `key_passes_per_90` | Passes cles /90 | plus haut | 5 |  |  | Contribution creative |

Metriques additionnelles a considerer :
- `offensive_duels_per_90`
- `successful_def_actions_per_90`
- `progressive_runs_per_90`
- `deep_completions_per_90`

## Catalogue Des Metriques Disponibles

Ces cles de metriques sont deja utilisees par les profils de scoring actuels. A privilegier, sauf demande explicite d'une nouvelle metrique pendant l'atelier.

| Cle metrique | Libelle |
| --- | --- |
| `accelerations_per_90` | Accelerations /90 |
| `accurate_crosses_percent` | Precision des centres (%) |
| `accurate_forward_passes_percent` | Precision passes vers l'avant (%) |
| `accurate_long_passes_percent` | Precision passes longues (%) |
| `accurate_passes_percent` | Precision de passe (%) |
| `accurate_progressive_passes_percent` | Precision passes progressives (%) |
| `aerial_duels_gk_per_90` | Duels aeriens GK /90 |
| `aerial_duels_per_90` | Duels aeriens /90 |
| `aerial_duels_won_percent` | Duels aeriens gagnes (%) |
| `assists_per_90` | Passes decisives /90 |
| `avg_pass_length_m` | Longueur moyenne de passe (m) |
| `back_passes_to_gk_per_90` | Passes en retrait au gardien /90 |
| `clean_sheets` | Clean sheets |
| `crosses_per_90` | Centres /90 |
| `deep_completions_per_90` | Deep completions /90 |
| `deep_crosses_per_90` | Centres profonds /90 |
| `def_duels_per_90` | Duels defensifs /90 |
| `def_duels_won_percent` | Duels defensifs gagnes (%) |
| `dribbles_per_90` | Dribbles /90 |
| `exits_per_90` | Sorties /90 |
| `forward_passes_per_90` | Passes vers l'avant /90 |
| `fouls_per_90` | Fautes /90 |
| `goal_conversion_rate` | Taux de conversion (%) |
| `goals_conceded_per_90` | Buts encaisses /90 |
| `goals_per_90` | Buts /90 |
| `goals_prevented_per_90` | Buts evites /90 |
| `interceptions_per_90` | Interceptions /90 |
| `key_passes_per_90` | Passes cles /90 |
| `lateral_passes_per_90` | Passes laterales /90 |
| `long_passes_per_90` | Passes longues /90 |
| `long_passes_received_per_90` | Longues passes recues /90 |
| `offensive_duels_per_90` | Duels offensifs /90 |
| `passes_per_90` | Passes /90 |
| `passes_received_per_90` | Passes recues /90 |
| `passes_to_final_third_per_90` | Passes vers dernier tiers /90 |
| `passes_to_penalty_area_per_90` | Passes vers surface /90 |
| `progressive_passes_per_90` | Passes progressives /90 |
| `progressive_runs_per_90` | Courses progressives /90 |
| `save_percent` | Arrets (%) |
| `shot_assists_per_90` | Passes menant a un tir /90 |
| `shots_against_per_90` | Tirs subis /90 |
| `shots_on_target_percent` | Tirs cadres (%) |
| `shots_per_90` | Tirs /90 |
| `sliding_tackles_per_90` | Tacles glisses /90 |
| `smart_passes_per_90` | Smart passes /90 |
| `successful_def_actions_per_90` | Actions defensives reussies /90 |
| `successful_dribbles_percent` | Dribbles reussis (%) |
| `through_passes_per_90` | Passes en profondeur /90 |
| `touches_in_penalty_area_per_90` | Touches dans la surface /90 |
| `xa_per_90` | xA /90 |
| `xg_against_per_90` | xGA /90 |
| `xg_per_90` | xG /90 |

## Template De Sortie Atelier

Remplir cette section apres l'atelier. C'est la partie que Codex convertira en configuration finale.

```yaml
scoring_model_version: "position_groups_v1"

global_weights:
  position_metrics: 75
  minutes_reliability: 10
  league_strength: 10
  team_strength: 5

minutes_policy:
  minimum_visible_minutes: 90
  stable_sample_minutes: 900
  full_confidence_minutes: 1800
  end_season_reference_minutes: 3000
  bonus_above_full_confidence: false

team_strength_policy:
  direction: "a_definir"
  notes: ""

position_groups:
  centre_backs:
    display_name: "Defenseurs centraux"
    positions: ["CB", "LCB", "RCB"]
    lower_is_better: []
    metrics:
      def_duels_won_percent: 16
      successful_def_actions_per_90: 12
```

## Questions Ouvertes Pour L'Atelier

- Les cotes gauche et droit doivent-ils utiliser un scoring identique, ou faut-il garder des modeles specifiques par cote ?
- Les defenseurs centraux excentres doivent-ils disparaitre totalement dans `Defenseurs centraux`, ou seulement rester comme filtres/tags ?
- Les lateraux et pistons doivent-ils etre groupes ensemble comme propose, ou separes ?
- Les milieux offensifs excentres (`LAMF`, `RAMF`) doivent-ils rester avec les milieux offensifs ou rejoindre les ailiers ?
- `goals_per_90` doit-il peser moins que `xg_per_90` pour reduire le bruit de finition ?
- Les metriques de discipline doivent-elles etre incluses globalement pour les postes defensifs ?
- Comment doit fonctionner la force d'equipe : booster la performance dans une equipe forte, compenser le contexte d'une equipe faible, ou seulement corriger les extremes ?
