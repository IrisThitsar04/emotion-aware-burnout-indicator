import streamlit as st
from inference import predict_full

#configure streamlit page
st.set_page_config(
    page_title="Burnout Indicator Detection Prototype",
    layout="wide",
)

st.title("Burnout Indicator Detection Prototype")

st.write(
    "This prototype applies a fine-tuned transformer model to user-provided text,"
    "identifies underlying emotions, aggregates them into affective quadrants, "
    "and generates a non-diagnostic burnout indicator. It is a conceptual, literature-based "
    "research demonstration intended solely as a supportive tool to illustrate how such "
    "a model can be integrated into a web interface. It is not designed or validated "
    "for clinical or psychological diagnosis."
)

#user input text box
text=st.text_area("Input text", height=200, placeholder="Type or paste a diary text or reflective message here...")

#run analysis when the user clicks the button
if st.button("Analyse") and text.strip():
  with st.spinner("Running the model..."):
    results=predict_full(text)
  res=results[0]

  #burnout indicator
  st.subheader("Burnout Indicator Output")
  st.success(res["burnout_indicator"])

  #quadrant probabilities
  st.subheader("Affective Quadrant Profile")
  qp=res["quadrant_probs"]

  st.write({
      "neutral_ambiguous": round(qp.get("neutral_ambiguous", 0.0),3),
      "pleasant_active": round(qp.get("pleasant_active", 0.0),3),
      "pleasant_deactive": round(qp.get("pleasant_deactive", 0.0),3),
      "unpleasant_active": round(qp.get("unpleasant_active", 0.0),3),
      "unpleasant_deactive": round(qp.get("unpleasant_deactive", 0.0),3),
  })

  #top 3 detected emotions
  st.subheader("Top 3 Detected Emotions")

  #sort all predicted emotions by probability
  emotions_sorted=sorted(res["emotions"], key=lambda x: x["p"], reverse=True)

  #keep only the three highest-confidence emtions
  emotions=emotions_sorted[:3]

  if emotions:
    table_rows=[
        {
            "Emotion":e["label"],
            "Predicted Probability":round(e["p"],3),
        }
        for e in emotions
    ]
    st.table(table_rows)
  else:
    st.info("No emotions detected")
else:
  st.info("Enter text and click 'Analyse' to see the results.")
