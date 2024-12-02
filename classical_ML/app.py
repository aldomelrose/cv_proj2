import streamlit as st
from models_inference import SignLanguageInference
import cv2

def main():
    st.title("Sign Language Translator")

    uploaded_file = st.file_uploader("Upload your model file", type="joblib")
    is_xgboost = st.checkbox("Is this an XGBoost model?", value=False)

    if st.button("Run Translator"):
        if uploaded_file is not None:
            model_path = uploaded_file.name
            with open(model_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            try:
                inferencer = SignLanguageInference(model_path, is_xgboost=is_xgboost)
                stframe = st.empty()
                letter_placeholder = st.empty()
                cap = cv2.VideoCapture(1)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame, final_prediction, avg_confidence = inferencer.run_inference_on_frame(frame)
                    stframe.image(frame, channels="BGR")

                    if final_prediction and avg_confidence >= inferencer.confidence_threshold:
                        letter_placeholder.markdown(f"### Recognized Letter: {final_prediction}")

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()
            except FileNotFoundError:
                st.error("Error: Model file not found!")
                st.write("Please check the path and make sure you have trained the model.")
                st.write(f"Looking for model at: {model_path}")
        else:
            st.error("Please upload a model file to proceed.")

if __name__ == "__main__":
    main()