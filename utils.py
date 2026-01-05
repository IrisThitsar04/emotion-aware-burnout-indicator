
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, precision_recall_curve, average_precision_score, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#define emotion groupings based on valence-arousal categories
pleasant_active = ['joy', 'love', 'admiration', 'amusement', 'excitement', 'pride', 'approval', 'desire']
pleasant_deactive = ['gratitude', 'relief', 'caring', 'optimism']
unpleasant_active = ['anger', 'annoyance', 'fear', 'disgust', 'nervousness', 'disapproval']
unpleasant_deactive = ['sadness', 'disappointment', 'remorse', 'embarrassment', 'grief']
neutral_ambiguous = ['neutral', 'surprise', 'curiosity', 'realization', 'confusion']

#map an emotion to its valence–arousal category
def categorize_emotion(emotion):
    if emotion in pleasant_active:
        return 'pleasant_active'
    elif emotion in pleasant_deactive:
        return 'pleasant_deactive'
    elif emotion in unpleasant_active:
        return 'unpleasant_active'
    elif emotion in unpleasant_deactive:
        return 'unpleasant_deactive'
    else:
        return 'neutral_ambiguous'


#aggregate emotion-level probabilities into category-level probabilities
def probs_emo_to_cat(probs_emo, categories, cat_to_emoIds, mode="noisy_or"):
  n, c=probs_emo.shape[0], len(categories)
  out=np.zeros((n,c), dtype=np.float32)
  for j, cat in enumerate(categories):
    idxs=cat_to_emoIds[cat]

    if len(idxs)==0:
      out[:,j]=0.0
    elif len(idxs)==1:
      out[:,j]=probs_emo[:,idxs[0]]
    elif mode=="noisy_or":
      out[:,j]=1.0-np.prod(1.0-probs_emo[:, idxs], axis=1)
    elif mode=="max":
      out[:,j]=probs_emo[:, idxs].max(axis=1)
    else:
      raise ValueError("Aggregation mode must be either 'noisy or' or 'max'")
  return out

#fixed 0.5 threshold baseline
def baseline_thresholds(y_val):
  num_labels = y_val.shape[1]
  return np.full(num_labels, 0.5)

#per label threshold optimisation using validation F1
def best_thresholds_per_label(val_scores, y_val, n_grid=50, use_quantiles=True, scores_are_logits=True):

  if scores_are_logits:
    val_scores= 1 / (1 + np.exp(-val_scores))

  num_labels = y_val.shape[1]
  best_ts = np.zeros(num_labels)


  for e in range(num_labels):
    scores_e = val_scores[:, e] #take all predicted values for label (e)
    y_true_e = y_val[:, e]      #take all actual values for label (e)

    if use_quantiles:
      q = np.linspace(0.02, 0.98, n_grid)
      candidcate_ts=np.quantile(scores_e, q) #get candidate thresholds from score quantiles
    else:
      candidcate_ts=np.linspace(scores_e.min(), scores_e.max(), n_grid)

    best_f1, best_t=-1.0, 0.5

    #evaluate each candidate threshold by its resulting F1 score
    for t in np.unique(candidcate_ts):
      y_pred_e = (scores_e >= t).astype(int)
      f1 = f1_score(y_true_e, y_pred_e, zero_division=0)

      if f1 > best_f1:
        best_f1, best_t = f1, t
    best_ts[e] = best_t
  return best_ts


#per category threshold tuning
def best_thresholds_per_category(val_scores, y_val, emo_to_cat, n_grid=50, use_quantiles=True, scores_are_logits=True):

  if scores_are_logits:
    val_scores= 1 / (1 + np.exp(-val_scores))

  num_labels = y_val.shape[1]
  categories = sorted(set(emo_to_cat))
  best_ts_cat = np.zeros(num_labels)

  for category in categories:
    emotion_id_cat=[i for i, c in enumerate(emo_to_cat) if c == category]
    scores_cat=val_scores[:, emotion_id_cat]
    y_true_cat=y_val[:, emotion_id_cat]

    flat_scores=scores_cat.reshape(-1)

    if use_quantiles:
      q=np.linspace(0.02, 0.98, n_grid)
      candidate_ts=np.quantile(flat_scores,q)
    else:
      candidate_ts=np.linspace(flat_scores.min(), flat_scores.max(), n_grid)

    best_f1, best_t=-1.0, 0.5

    for t in np.unique(candidate_ts):
      y_pred_cat=(scores_cat>=t).astype(int)

      f1s=[]
      for c in range (y_true_cat.shape[1]):
        if y_true_cat[:, c].sum()==0:
          continue
        f1s.append(f1_score(y_true_cat[:, c], y_pred_cat[:, c], zero_division=0))

        cat_macro_f1=np.mean(f1s) if len(f1s) else 0

        if cat_macro_f1>best_f1:
          best_f1, best_t=cat_macro_f1, t

    for c in emotion_id_cat:
      best_ts_cat[c]=best_t
  return best_ts_cat


#blend per-emotion and per-category thresholds using label frequency
def best_thresholds_blended(val_scores, y_val, emo_to_cat, k=50, n_grid=50, use_quantiles=True, scores_are_logits=True, clamp_quantiles=(0.05, 0.95)):

  if scores_are_logits:
    val_scores= 1 / (1 + np.exp(-val_scores))

  ts_per_emotion=best_thresholds_per_label(val_scores, y_val, n_grid, use_quantiles, scores_are_logits=True) #per-emotion thresholds

  ts_per_category=best_thresholds_per_category(val_scores, y_val, emo_to_cat, n_grid, use_quantiles) #per-category thresholds

  num_labels=y_val.shape[1]
  ts_blended=ts_per_emotion.copy()

  pos_counts=(y_val==1).sum(axis=0).astype(float)

  alpha=pos_counts/(pos_counts+k)
  ts_blended=alpha*ts_per_emotion+(1.0-alpha)*ts_per_category #blend weight (frequent labels use per-emotion threshold, rare labels use per-category threshold)

  #clamp each emotion threshold
  if clamp_quantiles is not None:
    q_low, q_high=clamp_quantiles
    for e in range(num_labels):
      low, high=np.quantile(val_scores[:,e], [q_low, q_high])
      ts_blended[e]=float(np.clip(ts_blended[e], low, high))

    return ts_blended

#evaluate predictions and report metrics
def evaluate_run(y_true, y_scores, thresholds, label_names=None, title="Evaluation Results", return_metrics=False):

    #apply thresholds to convert scores into binary predictions
    y_pred = (y_scores >= thresholds).astype(int)

    print(f"\n{title}")

    #compute evaluation metrics
    micro_f1=f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro_f1=f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_precision=precision_score(y_true, y_pred, average='micro', zero_division=0)
    micro_recall=recall_score(y_true, y_pred, average='micro', zero_division=0)

    print(f"Micro F1: {micro_f1:.2f}")
    print(f"Macro F1: {macro_f1:.2f}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))

    if return_metrics==True:
      metrics={
          "micro-f1":micro_f1,
          "macro-f1":macro_f1,
          "micro-precision":micro_precision,
          "micro-recall":micro_recall,
          "per_label_f1":f1_score(y_true, y_pred, average=None, zero_division=0),
          "per_label_precision":precision_score(y_true, y_pred, average=None, zero_division=0),
          "per_label_recall":recall_score(y_true, y_pred, average=None, zero_division=0)
      }

      return metrics

#compute per-label confusion matrices
def confusion_matrices_per_label(y_true, y_scores, thresholds, label_names=None):
  y_pred = (y_scores >= thresholds).astype(int)

  #compute confusion matrices for each label
  mcm=multilabel_confusion_matrix(y_true, y_pred)

  num_labels=mcm.shape[0]
  if label_names is None:
    label_names=[f"Label {i}" for i in range(num_labels)]
  cm_dict={}
  for i, name in enumerate(label_names):
    TN, FP, FN, TP=mcm[i].ravel()
    cm_dict[name]={
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "TP": TP,
        "matrix":mcm[i]
    }
  return cm_dict

#plot validation and test confusion matrices
def plot_cm(cm_val_dict, cm_test_dict, label_names, normalize=False):

  num_labels=len(label_names)

  fig, axes=plt.subplots(
      nrows=2, ncols=num_labels,
      figsize=(4*num_labels, 8)
  )

  if num_labels==1:
    axes=np.array([[axes[0]], [axes[1]]])

  for row_idx, (cm_dict, row_title) in enumerate([
      (cm_val_dict, "Validation"),
      (cm_test_dict, "Test")
  ]):
    for col_idx, label in enumerate(label_names):

      ax=axes[row_idx, col_idx]
      cm=np.asarray(cm_dict[label]["matrix"])
      if normalize:
        total=cm.sum()
        cm_plot=cm/total if total>0 else cm.astype(float)
      else:
        cm_plot=cm

      sns.heatmap(cm_plot, annot=True, fmt=".2f", cmap="Blues", cbar=False, xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"], ax=ax)

      ax.set_title(f"\n\n{row_title} : {label}\n")
      ax.set_ylabel("Actual")
      ax.set_xlabel("Predicted")

  plt.tight_layout()
  plt.show()

#plot validation vs test micro and macro F1
def plot_val_test_macro_micro(val_metrics, test_metrics, model_name="Model"):
  labels=["Micro-F1", "Macro-F1"]
  val_scores=[val_metrics["micro-f1"], val_metrics["macro-f1"]]
  test_scores=[test_metrics["micro-f1"], test_metrics["macro-f1"]]

  x=np.arange(len(labels))
  width=0.35

  plt.figure(figsize=(10,6))
  plt.bar(x-width/2, val_scores, width, label="Validation", color="powderblue")
  plt.bar(x+width/2, test_scores, width, label="Test")

  plt.xticks(x, labels)
  plt.ylim(0,1.0)
  plt.ylabel("Score")
  plt.title(f"{model_name}: Validation vs Test (Macro vs Micro F1)")
  plt.legend()
  plt.tight_layout()
  plt.show()

#plot per-category F1 comparison between val and test splits
def plot_val_test_per_cat_f1(val_metrics, test_metrics, categories, model_name="Model"):
  categories=list(categories)

  val_f1_per_cat  = val_metrics["per_label_f1"]
  test_f1_per_cat = test_metrics["per_label_f1"]

  x = np.arange(len(categories))
  width = 0.35

  plt.figure(figsize=(10,6))
  plt.bar(x-width/2, val_f1_per_cat, width, label="Validation", color="lightgray")
  plt.bar(x+width/2, test_f1_per_cat, width, label="Test")

  plt.xticks(x, categories, rotation=20, ha='right')
  plt.ylim(0,1.0)
  plt.ylabel("F1-Score")
  plt.title(f"{model_name}: Validation vs Test (Per Category F1)")
  plt.legend()
  plt.tight_layout()
  plt.show()

#plot precision-recall curves per category
def plot_pr_curves_per_cat(y_test_cat, test_scores_cat, categories, model_name="Model"):
  categories=list(categories)
  plt.figure(figsize=(10,6))

  for j, cat in enumerate(categories):
    y_true_cat  = y_test_cat[:, j]
    y_score_cat = test_scores_cat[:, j]

    precision, recall, _ = precision_recall_curve(y_true_cat, y_score_cat)
    avg_precision = average_precision_score(y_true_cat, y_score_cat)

    plt.plot(recall, precision, label=f"{cat} (AP = {avg_precision:.2f})")

  plt.xlabel("Recall")
  plt.ylabel("Precision")
  plt.title(f"{model_name}: Precision–Recall Curves per Category (Test Split)")
  plt.tight_layout()
  plt.legend()
  plt.show()

#convert emotion-level labels to category-level labels
def y_to_categories(y, categories, cat_to_emoIds):
  y_cat=np.zeros((y.shape[0], len(categories)), dtype=int)
  for j, c in enumerate(categories):
    y_cat[:, j]=(y[:, cat_to_emoIds[c]].sum(axis=1)>0).astype(int)
  return y_cat
