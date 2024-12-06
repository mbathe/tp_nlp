from bert_score import BERTScorer



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

    prec = precision(tp, fp)
    rec = recall(tp, fn)
    f1 = f_measure(prec, rec)
    spec = specificity(tn, fp)
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


def get_bert_score(reference, candidate):
    """
    Calculate the BERT score between the resume document and the original document.

    Parameters:
    -----------
    resume_doc : str
        The resume document to compare.
    doc : str
        The original document to compare.

    Returns:
    --------
    bert_score : float
        The BERT score between the resume document and the original document.

    Side Effects:
    ------------
    None
    """
    scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, F1 = scorer.score(candidate, reference)
    return P, R, F1
