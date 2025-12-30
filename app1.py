# app.py
import os
import streamlit as st
from dotenv import load_dotenv

# LangChain / agent imports (your project previously used these)
from langchain_groq import ChatGroq
from langchain_classic.chains import LLMMathChain, LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents import Tool, initialize_agent
from langchain_classic.callbacks import StreamlitCallbackHandler

# Load .env (optional)
load_dotenv()

# -------------------------
# Streamlit UI config
# -------------------------
st.set_page_config(
    page_title="Smart Math Solver",
    page_icon="assets/mathlogo.png" )

st.title("Word Problem Solver — step-by-step reasoning (agent demo)")

# Sidebar for API key
groq_api_key = st.sidebar.text_input(label="Groq API KEY", type="password")
if not groq_api_key:
    st.info("Enter your Groq API Key in the sidebar to continue.")
    st.stop()

# -------------------------
# LLM and tools initialization
# -------------------------
# NOTE: Provide the correct model name required by your provider if necessary.
# If ChatGroq expects a model string, put it here. If not, leave empty as required.
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

# Wikipedia tool (optional)
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Use to fetch short factual information when needed."
)

# Math chain / calculator tool
math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Performs arithmetic calculations. Input only arithmetic expressions or numbers."
)

# -------------------------
# Prompt template that forces step-by-step reasoning
# -------------------------
prompt_text = """
You are a careful math tutor. For the question below:

1. Show **clear, numbered steps** (Step 1:, Step 2:, ...). Each step must include the short reasoning and the intermediate numeric result.
2. When you need arithmetic, call the Calculator tool and **explicitly** include the calculator result in the step text (e.g., "Calculation: 240 * 3/5 = 144").
3. After all steps, produce a **Final Answer:** line showing the numeric answer.
4. Do NOT hide reasoning — be explicit and stepwise.

Question: {question}

Answer:
"""

prompt_template = PromptTemplate(input_variables=["question"], template=prompt_text)
reasoning_chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="ReasoningTool",
    func=reasoning_chain.run,
    description="Produces a step-by-step explanation for word problems using the Calculator tool results."
)

# -------------------------
# Initialize the agent (verbose to show intermediate steps)
# -------------------------
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,                 # show agent logs and intermediate tool usage
    handle_parsing_error=True
)

# -------------------------
# Session state and chat UI
# -------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi — I will solve math word problems step-by-step. Paste your question below."}
    ]

# Render existing messages
for m in st.session_state["messages"]:
    st.chat_message(m["role"]).write(m["content"])

# Input area
question = st.text_area("Enter your math problem here:", height=140)

# Button to trigger
if st.button("Find my answer"):
    if not question or not question.strip():
        st.warning("Please enter a math question.")
    else:
        # Append user message
        st.session_state["messages"].append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        # Prepare Streamlit callback handler to stream agent/tool events into UI
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        # Run the agent — give the raw question (agent will use reasoning tool and calculator)
        try:
            with st.spinner("Thinking and running tools..."):
                # Using assistant_agent.run streams events to callback handler when verbose=True
                agent_response = assistant_agent.run(question, callbacks=[st_cb])

            # Some agent runtimes return dict or object. Ensure string for display.
            if not isinstance(agent_response, str):
                agent_text = str(agent_response)
            else:
                agent_text = agent_response

            # Append assistant reply and display
            st.session_state["messages"].append({"role": "assistant", "content": agent_text})
            st.chat_message("assistant").write(agent_text)

        except Exception as e:
            err_msg = f"Error while generating response: {type(e).__name__}: {e}"
            st.session_state["messages"].append({"role": "assistant", "content": err_msg})
            st.chat_message("assistant").write(err_msg)

# Small helpful examples to paste
st.markdown("---")
st.markdown("**Example questions (click to copy):**")
if st.button("Copy example 1"):
    st.session_state["example_text"] = "A shopkeeper had 240 candies. He sold 3/5 of them in the morning and then bought 48 more in the afternoon. In the evening, he sold half of the remaining candies. How many candies does he have left now? Explain the steps clearly."
if st.button("Copy example 2"):
    st.session_state["example_text"] = "A father is three times as old as his son. In 12 years, the father will be twice as old as the son. What are their current ages? Explain your reasoning step-by-step."

# Put example into the text area if the user clicked a button
if "example_text" in st.session_state and st.session_state["example_text"]:
    st.experimental_set_query_params()  # no-op but inside to visually update UI
    # show example below the input to encourage pasting manually

st.caption("Tip: if you still get only a short numeric answer, try increasing the prompt strictness (request more steps) or check the model parameters your provider supports.")

