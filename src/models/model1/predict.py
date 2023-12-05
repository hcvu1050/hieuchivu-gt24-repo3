from ...constants import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import precision_recall_curve
def get_metrics (model, cv_dataset):
    """return the performance metrics of a model given validation data.\n

    Args:
        model (keras.Model): a model instance
        cv_dataset (tf.data.Dataset): crosvalidation dataset

    Returns:
        _type_: best threshold (for F1 score), best F1 score, AUC PR
    """
    batch_cv_dataset =cv_dataset.batch(32)
    val_true = np.array ([label for _, label in cv_dataset])
    val_pred_logit = model.predict (batch_cv_dataset).flatten()
    val_pred_prob = tf.keras.activations.sigmoid(val_pred_logit)
    
    m = tf.keras.metrics.AUC(num_thresholds=200, curve = 'PR', from_logits = True)
    m.update_state (val_true, val_pred_logit, sample_weight = None)
    auc_pr = m.result().numpy()
    
    precisions, recalls, thresholds = precision_recall_curve(val_true, val_pred_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_threshold_index = np.argmax(f1_scores)
    
    best_threshold = thresholds[best_threshold_index]
    best_f1_score = f1_scores[best_threshold_index]
    result = {
        'best_threshold': best_threshold,
        'best_f1_score' : best_f1_score,
        'auc_pr': auc_pr,
    }
    
    return result