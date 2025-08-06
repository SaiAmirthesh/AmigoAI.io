import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os
from datetime import datetime
import json

load_dotenv()

st.set_page_config(
    page_title="Amigo AI Assistant",
    page_icon="🤖", 
    layout="centered", 
    initial_sidebar_state="expanded" 
)

st.title("🤝 Amigo AI Assistant")
st.markdown("---")

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("🚨 GOOGLE_API_KEY environment variable not set. Please set it in your `.env` file to enable the chatbot functionality.")
    st.stop()

@tool
def calculator_add(a: float, b: float) -> str:
    print(f"Tool 'calculator' called with a={a}, b={b}")
    return f"The sum of {a} and {b} is {a + b}"

@tool
def calculator_sub(a: float, b: float) -> str:
    print(f"Tool 'calculator' called with a={a}, b={b}")
    return f"The sum of {a} and {b} is {a - b}"

@tool
def calculator_mul(a: float, b: float) -> str:
    print(f"Tool 'calculator' called with a={a}, b={b}")
    return f"The sum of {a} and {b} is {a * b}"

@tool
def calculator_div(a: float, b: float) -> str:
    print(f"Tool 'calculator' called with a={a}, b={b}")
    return f"The sum of {a} and {b} is {a / b}"

@tool
def calculator_mod(a: float, b: float) -> str:
    print(f"Tool 'calculator' called with a={a}, b={b}")
    return f"The sum of {a} and {b} is {a % b}"

@tool
def say_hello(name: str) -> str:
    print(f"Tool 'say_hello' called with name='{name}'")
    return f"Hello {name}, I hope you are well today"

@tool
def get_current_datetime() -> str:
    print("Tool 'get_current_datetime' called.")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def reverse_string(text: str) -> str:
    print(f"Tool 'reverse_string' called with text='{text}'")
    return text[::-1]

@tool
def is_even_or_odd(number: int) -> str:
    print(f"Tool 'is_even_or_odd' called with number={number}")
    if number % 2 == 0:
        return f"The number {number} is even."
    else:
        return f"The number {number} is odd."

@st.cache_resource
def initialize_agent():
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"))
    tools = [calculator_add,calculator_sub,calculator_mul,calculator_div,calculator_mod, say_hello, get_current_datetime, reverse_string, is_even_or_odd]
    agent_executor = create_react_agent(model, tools)
    return agent_executor

if google_api_key:
    agent_executor = initialize_agent()
else:
    agent_executor = None 

st.sidebar.header("About Amigo AI 🤖")
st.sidebar.write(
    "Your friendly AI assistant, powered by Google Gemini and LangChain! "
    "I'm here to help you with a variety of tasks using my specialized tools."
)

st.sidebar.subheader("My Capabilities (Tools) 👇")
st.sidebar.markdown(
    """
    - **🔢 Calculator**: Performs basic arithmetic.
    - **👋 Greeter**: Says hello and wishes well.
    - **⏰ Date & Time**: Provides the current date and time.
    - **🔄 String Reverser**: Reverses any text you give me.
    - **❓ Even/Odd Checker**: Determines if a number is even or odd.
    """
)


st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Try asking me something like 'What is 15 times 7?' or 'Reverse the word 'Streamlit''.")


if "messages" not in st.session_state:
    st.session_state.messages = []
    initial_message = "Hello there, Amigo! I'm your AI assistant. I can help with calculations (e.g., 'What is 5 + 3?'), greetings (e.g., 'Say hello to John'), getting the current time, reversing text, and checking if numbers are even or odd. How can I help you today?"
    if not google_api_key:
        initial_message = "Please set your GOOGLE_API_KEY in the .env file to enable the chatbot functionality."
    st.session_state.messages.append({"role": "assistant", "content": initial_message})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if google_api_key:
    user_input = st.chat_input("Type your message here...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                for chunk in agent_executor.stream(
                    {"messages": [HumanMessage(content=user_input)]}
                ):
                    if "agent" in chunk and "messages" in chunk["agent"]:
                        for message in chunk["agent"]["messages"]:
                            full_response += message.content
                            message_placeholder.write(full_response + "▌") 
                message_placeholder.write(full_response) 
            except Exception as e:
                st.error(f"An error occurred: {e}")
                full_response = "I apologize, but I encountered an error while processing your request. Please try again or rephrase your question."

        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.chat_input("Chatbot is disabled. Please set your API key.", disabled=True)


st.markdown("---")
