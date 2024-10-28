def calculate_metrics(relevant_indices, retrieved_indices):
    """
    Calcule les métriques à partir des indices des documents pertinents et retournés.

    Parameters:
    relevant_indices (list): Une liste d'indices représentant les documents pertinents.
    retrieved_indices (list): Une liste d'indices représentant les documents retournés par le système.

    Returns:
    dict: Un dictionnaire contenant les valeurs des métriques calculées. Les clés sont les noms des métriques,
          et les valeurs sont les valeurs numériques correspondantes.
    """
    relevant_set = set(relevant_indices)
    retrieved_set = set(retrieved_indices)

    tp = len(relevant_set.intersection(retrieved_set))  # True Positives
    fp = len(retrieved_set - relevant_set)               # False Positives
    fn = len(relevant_set - retrieved_set)               # False Negatives
    tn = 0  # Supposons que nous n'avons pas d'informations sur les vrais négatifs

    # Calcul des métriques
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    f1 = f_measure(prec, rec)
    spec = specificity(tn, fp)  # Spécificité n'est pas calculable sans TN
    fpr = false_positive_rate(fp, tn)

    return {
        "Précision": prec,
        "Rappel": rec,
        "F-mesure": f1,
        "Spécificité": spec,
        "Taux de faux positifs": fpr,
    }


def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f_measure(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def specificity(tn, fp):
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def false_positive_rate(fp, tn):
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0

# Exemple d'utilisation


def main():
    # Indices des documents pertinents et retournés
    relevant_indices = [1, 2, 3, 4, 5]  # Exemples d'indices pertinents
    retrieved_indices = [3, 4, 5, 6, 7]  # Exemples d'indices retournés

    metrics = calculate_metrics(relevant_indices, retrieved_indices)

    # Affichage des résultats
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
