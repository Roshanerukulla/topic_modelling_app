# streamlit_app.py

import streamlit as st
from ModelTrainer import ModelTrainer
import os 


class StreamlitApp:
    def __init__(self, model_trainer):
        self.model_trainer = model_trainer

    def _display_comments(self, query):
        # Your function to display comments related to the entered keyword
        for idx, row in self.model_trainer.df.iterrows():
            if query.lower() in row['Text'].lower():
                st.write(f"Username: {row['ProfileName']}")
                st.write(f"Original Comment: {row['OriginalComment']}")  # Display the original comment
                st.write("----")  # Adding a separator for clarity

    def _display_lda_topics(self, query):
        # Your function to display LDA topics related to the entered keyword
        lda_topics = self.model_trainer.get_lda_topics(query)
        st.write(f"LDA Topics related to the entered keyword '{query}':")
        for topic, prob in lda_topics:
            st.write(f"Topic {topic}: Probability - {prob:.4f}")

    def run_app(self):
        st.title("Topic Search App")
        query = st.text_input("Enter keyword to search comments by topic:")

        if query:
            st.subheader("Comments related to the entered keyword:")
            self._display_comments(query)

            st.subheader("LDA Topics related to the entered keyword:")
            self._display_lda_topics(query)


if __name__ == "__main__":
    # Training the model
    model_trainer = ModelTrainer("./Reviews.csv")

    # Creating the Streamlit app
    streamlit_app = StreamlitApp(model_trainer)

    # Running the Streamlit app
    streamlit_app.run_app()
