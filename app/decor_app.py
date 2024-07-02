import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent,Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
from dotenv import load_dotenv
from lyzr_automata.tasks.task_literals import InputType, OutputType
import os

load_dotenv()
api = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="Lyzr Interior Designer",
    layout="centered",  # or "wide"
    initial_sidebar_state="auto",
    page_icon="lyzr-logo-cut.png",
)

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Lyzr Interior Designer")
st.sidebar.markdown("## Welcome to the Lyzr Interior Designer!")
st.sidebar.markdown("This App Harnesses power of Lyzr Automata to generate Interior Design.User Needs to input Style,Room And Instructions and This app generates Interior Design.")

style = st.sidebar.selectbox("Style", ("Modern","Minimalistic","Tropical","Coastal","Futuristic","Cyberpunk"))
room = st.sidebar.selectbox("Room", ("Living Room","Bedroom","Bathroom","Outdoor Garden","House Exterior"))
instruction = st.sidebar.text_area("Enter your Instruction: ")


def generate_image(style, room_type, insights):
    artist_agent = Agent(
        prompt_persona="You are an Interior Designer.Your Task Is to generate Interior from given Instructions.",
        role="Artist",
    )

    open_ai_model_image = OpenAIModel(
        api_key=api,
        parameters={
            "n": 1,
            "model": "dall-e-3",
        },
    )

    interior_task = Task(
        name="Art Image Creation",
        output_type=OutputType.IMAGE,
        input_type=InputType.TEXT,
        model=open_ai_model_image,
        agent=artist_agent,
        log_output=True,
        instructions=f"""Your Task Is To Generate Interior Design.

            Follow Below Instructions for Interior Design:
            {style} style ({room_type}) with {insights}
            """,
    )

    output = LinearSyncPipeline(
        name="Generate Interior",
        completion_message="Interior Generated!",
        tasks=[
            interior_task
        ],
    ).run()
    return output[0]['task_output'].url


if st.sidebar.button("Generate"):
    solution = generate_image(style, room, instruction)
    st.image(solution)
