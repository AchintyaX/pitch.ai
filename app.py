import pandas as pd 
import os 
import openai 
import streamlit as st 
from loguru import logger
import re 
from search_utils import PitchCollection
import random 
from ast import literal_eval
class Application:
    def __init__(self) -> None:
        self.PitchObj = PitchCollection()
    

    def run(self):
        """
        Run the streamlit application 
        """
        st.title("Pitch.ai")
        st.markdown("A Utility for generalists to find pitches from PR firms related to their stories")

        input_str = st.text_input("Enter Description")
        input_keywords = st.text_input("Enter Keywords")

        

        if st.button("Get Matches"):

            if input_str and input_keywords:
                results = self.PitchObj([input_str], input_keywords)
                st.write(pd.DataFrame(results))  
            


@st.cache_resource
def create():
    """
    Creates and caches a Streamlit application.
    Returns:
        Application
    """

    return Application()


if __name__ == "__main__":
    # Create and run application
    app = create()
    app.run()