"""
app.py
"""
# Standard imports
import base64
import json
import os

# Third party imports
import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field

# Constants
SYSTEM_PROMPT = """
You are a world-class chemistry teacher.
You are given an image of a student's work containing chemical formulas.
Your task is to extract the chemical formulas from the image. Reply in LaTeX
Do NOT fix any errors as this is for grading purposes.
Before you give the equations, returning `observation_and_reasoning` as the first key. The value for that first key is free text.
Reply in valid JSON without any other text, explanation or code blocks.
""".strip()

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.error("OPENAI_API_KEY is not set.")
        st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Constants
MODELS = {
    "4o": "gpt-4o-2024-08-06",
    "4o-mini": "gpt-4o-mini-2024-07-18",
    "4o-vision-ft": "ft:gpt-4o-2024-08-06:gabrielc:chem-diagrams:AURkfCfw",
}

# Pydantic models
class Equation(BaseModel):
    """A chemical equation."""
    LHS: str = Field(description="The left hand side of the equation. Can be an empty string if there is no equation.")
    RHS: str = Field(description="The right hand side of the equation. Can be an empty string if there is no equation.")

class ChemicalEquations(BaseModel):
    """ A collection of chemical equations. """
    observation_and_reasoning: str = Field(description="Observations about the diagram and your reasoning.")
    equation_1: Equation = Field(description="The first equation. The LHS and RHS can be empty if there is less than 1 equation.")
    equation_2: Equation = Field(description="The second equation. The LHS and RHS can be empty if there are less than 2 equations.")
    equation_3: Equation = Field(description="The third equation. The LHS and RHS can be empty if there are less than 3 equations.")
    equation_4: Equation = Field(description="The fourth equation. The LHS and RHS can be empty if there are less than 4 equations.")
    equation_5: Equation = Field(description="The fifth equation. The LHS and RHS can be empty if there are less than 5 equations.")

# Call GPT 4o to extract equations from image
def call_llm(image_base64, model="gpt-4o-2024-08-06"):
    """ Call LLM extract equations from image. """
    
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }
            ]
            },
        ],
        response_format=ChemicalEquations,
        temperature=0.0,
        seed=42,
        max_tokens=4000
    )

    parsed_equations = completion.choices[0].message.parsed

    return parsed_equations

def display_equations(equations):
    """Helper function to display equations consistently."""
    if not equations:
        st.write("No valid equations were extracted.")

    for i in range(1, 6):
        equation = getattr(equations, f"equation_{i}")
        if equation.LHS:
            st.write(f"Equation {i}:")
            st.latex(f"{equation.LHS} = {equation.RHS}")

### UI ###
# Streamlit App
st.set_page_config(layout="wide")
st.title("Chemistry Equation Extractor - Demo")

# File uploader
uploaded_file = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Convert the uploaded image to Base64
    image_bytes = uploaded_file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # Display image
    st.image(uploaded_file, caption="Uploaded Image", width=400)

    if st.button("Submit"):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.header("GPT 4o Mini")
            with st.spinner("Processing..."):
                equations = call_llm(image_base64, model=MODELS["4o-mini"])
                display_equations(equations)
                
        with col2:
            st.header("GPT 4o")
            with st.spinner("Processing..."):
                equations = call_llm(image_base64, model=MODELS["4o"])
                display_equations(equations)
                
        with col3:
            st.header("Vision Finetuned GPT 4o")
            with st.spinner("Processing..."):
                equations = call_llm(image_base64, model=MODELS["4o-vision-ft"])
                display_equations(equations)
                        
