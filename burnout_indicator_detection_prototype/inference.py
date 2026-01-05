
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import sys

#directory containing the fine-tuned model, tokenizer, and metadata
model_dir='best_emo_model'

device="cuda" if torch.cuda.is_available() else "cpu"

#load tokenizer and fine-tuned transformer model
tokenizer=AutoTokenizer.from_pretrained(model_dir)
model_final=AutoModelForSequenceClassification.from_pretrained(model_dir)
model_final.to(device)

#set model for inference
model_final.eval()

#load saved metadata for inference and interpretation
with open(os.path.join(model_dir, "meta_data.json"), "r") as f:
  meta_data=json.load(f)

emotion_labels=meta_data["emotion_labels"]
emo_thresholds=np.array(meta_data["emo_thresholds"])
categories=meta_data["categories"]
emo_to_cat=meta_data["emo_to_cat"]
cat_thresholds=np.array(meta_data["cat_thresholds"])
max_len=meta_data["max_len"]
aggregation=meta_data["aggregation"]

#sigmoid for multi-label probabilities
sigmoid=torch.nn.Sigmoid()

#ensure inputs are handled as a list of strings
def normalise_texts(texts):
  if isinstance(texts, str):
    return [texts]
  return [str(t) for t in texts]

@torch.no_grad()

#return proabability scores for each emotion label
def predict_emotions_prob(text):

  texts=normalise_texts(text)

  enc=tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_len).to(device)

  logits=model_final(**enc).logits
  probs=sigmoid(logits)
  return probs.cpu().numpy()

@torch.no_grad()

#return emotion probabilities and binary predictions
def get_emotion_predictions(text):

  probs=predict_emotions_prob(text)
  preds=(probs>=emo_thresholds).astype(int)

  all_results=[]
  for row_probs, row_preds in zip(probs, preds):
    res=[]
    for label_name, p, y in zip(emotion_labels, row_probs, row_preds):
      res.append({
          "label": label_name,
          "p": float(p),
          "y": int(y)
      })
    all_results.append(res)
  return all_results

#aggregate emotion probabilities into quadrant probabilities
def probs_emo_to_cat(probs_emo, emo_to_cat, categories, mode="noisy_or"):
  probs_emo=np.asarray(probs_emo)
  n=probs_emo.shape[0]
  c=len(categories)
  out=np.zeros((n,c), dtype=np.float32)

  cat_to_emoIds={cat: [i for i, c in enumerate(emo_to_cat) if c==cat] for cat in categories}

  for j, cat in enumerate(categories):
    idxs=cat_to_emoIds[cat]

    if len(idxs)==0:
      out[:,j]=0.0
    elif len(idxs)==1:
      out[:,j]=probs_emo[:,idxs[0]]
    else:
      out[:,j]=1.0-np.prod(1.0-probs_emo[:, idxs], axis=1)
  return out

#map quadrant predictions to burnout indicator
def burnout_indicator_from_quadrants(pred_row):

  col_idx={c:i for i, c in enumerate(categories)}

  NEU=int((pred_row[col_idx["neutral_ambiguous"]]))
  PA=int((pred_row[col_idx["pleasant_active"]]))
  PD=int((pred_row[col_idx["pleasant_deactive"]]))
  NA=int((pred_row[col_idx["unpleasant_active"]]))
  ND=int((pred_row[col_idx["unpleasant_deactive"]]))

  if ND:
    return "Signs of Advanced Burnout (Exhaustion/Ineffectiveness)"
  elif NA:
    return "Signs of Moderate Burnout (Stress/Cynicism)"
  elif PA:
    return ("Indicators of Engagement (No Apparent Signs of Burnout)")
  elif PD:
    return ("Indicators of Satisfaction (No Apparent Signs of Burnout)")
  elif NEU:
    return ("Ambiguous Burnout Indicator")
  else:
    return ("Ambiguous Burnout Indicator")

#full inference pipeline from text to burnout indicator
@torch.no_grad()
def predict_full(texts):
  probs_emo=predict_emotions_prob(texts)
  all_emotions=get_emotion_predictions(texts)
  probs_cat_all=probs_emo_to_cat(
      probs_emo, emo_to_cat, categories, aggregation
  )
  out=[]

  for row_probs_cat, emo_res in zip(probs_cat_all, all_emotions):
      preds_cat=(row_probs_cat>=cat_thresholds).astype(int)
      burnout_label=burnout_indicator_from_quadrants(preds_cat)
      out.append(
          {
              "emotions": emo_res,
              "quadrant_probs":{
                  cat: float(p) for cat, p in zip(categories, row_probs_cat)},
              "quadrant_preds":{
                  cat: int(y) for cat, y in zip(categories, preds_cat)},
              "burnout_indicator": burnout_label,
          },
      )
  return out
