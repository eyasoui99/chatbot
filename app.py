from dotenv import load_dotenv
import streamlit as st
import requests
import json
import os
from langdetect import detect
import google.generativeai as genai
import pandas as pd
faq = pd.read_csv('faq_questions_answers.csv')
# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="Bilingual Chatbot Agent",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_configured" not in st.session_state:
    st.session_state.api_configured = False

# Gemini configuration
def configure_gemini():
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            st.error("GEMINI_API_KEY not set in environment.")
            return False
        genai.configure(api_key=api_key)
        st.session_state.api_configured = True
        return True
    except Exception as e:
        st.error(f"Gemini configuration failed: {str(e)}")
        return False

# Language detection
def detect_language(text):
    try:
        lang = detect(text)
        return "French" if lang == "fr" else "English"
    except:
        return "English"

# Classify query using Gemini
def classify_query(query):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
        prompt = f"""
        You are a classifier assistant. Your task is to:
        1. Understand the user's query: "{query}" (it may be in French).
        2. Translate it to English if needed.
        3. Classify the **English version** of the query into **exactly one** of the following three labels:

        ---

        1. **text2sql** → Use this label if the query is about retrieving influencer-related data from a database. This includes:
        - Influencer details or profiles (e.g., influence themes, center of interest, email, country, etc.)
        - Instagram community or follower insights
        - Statistics, audience, or Instagram performance details
        - Sales, clicks, or conversion rates related to a specific influencer, brand, or product
        - Information about brands or products
        - Lists or rankings of products/brands in specific categories (e.g., "top 10 products in X"), possibly with conditions (e.g., location-based filters)

        2. **analyze** → Use this label if the query is about legal documents, explanations, platform-related information, or general help. This includes:
        - If the query {query}, when translated to French, matches any item in {faq['question']}, it should be categorized as analyze.
        - Privacy Policy: questions about user data usage, protection, or collection
        - Terms of Service (CGU): user rights and platform conditions
        - Platform help: how things work on Shop My Influence
        - Any query about influencer accounts or campaign conditions not asking for specific data
        - General platform usage or guidance

        3. **web** → Use this label for general web-based or external content not specific to influencer data or platform documentation. This includes:
        - News, current events, or market trends
        - Popular culture, general curiosity, or public info not tied to the platform
        - Greetings or non-informational content

        **Important**:
        - If the query is not clearly 'analyze' or 'web', and it relates to influencer data or analytics, **classify it as 'text2sql'**.
        - Return only one of the following: `text2sql`, `analyze`, or `web`.
        - Do not explain your reasoning or return anything else.
        """ 

        response = model.generate_content(prompt)
        label = response.text.strip().lower()
        print(label)
        return label if label in {"text2sql", "analyze", "web"} else "text2sql"
    except Exception as e:
        print(f"Gemini classification failed: {e}")
        return "text2sql"

# Unified API handler
def call_api(query, language, influencer_uid="la0NUVFtxnNnYng2JJF9i2FzkYz1"):
    query_type = classify_query(query)

    if query_type == "web":
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                f"Tu es un assistant qui répond à des questions d’actualité en français. Réponds clairement à cette question sans poser de questions supplémentaires : {query}"
                if language == "French"
                else f"You are a helpful assistant answering news/trend queries. Give a structured and concise answer without asking any follow-up questions: {query}"
            )
            response = model.generate_content(prompt)
            return {"success": True, "result": response.text}
        except Exception as e:
            return {"success": False, "error": f"Gemini error: {str(e)}"}

    url = (
        "https://text2sql-mffb.onrender.com/api/analyze"
        if query_type == "analyze"
        else "https://text2sql-mffb.onrender.com/api/query"
    )
    data = {
        "query": query,
        "influencer_uid": influencer_uid
    }

    try:
        response = requests.post(url, headers={"Content-Type": "application/json"}, json=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"success": False, "error": "API request timed out"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"API error: {str(e)}"}
    except json.JSONDecodeError:
        return {"success": False, "error": "Invalid JSON returned from API"}

# Generate response using Gemini
def generate_natural_response(api_response, user_query, language):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        if "references" in api_response:
            api_summary = f"""
Query: {api_response.get("query", "")}

Answer:
{api_response.get("answer", "")}

References:
{chr(10).join(api_response.get("references", []))}
"""
        else:
            api_summary = f"""
Query: {api_response.get("natural_language_query", "")}

Result:
{api_response.get("result", "").strip()}

Explanation:
{api_response.get("explanation", "")}
"""

        if language == "French":
            prompt = f"""
            Tu es un assistant virtuel utile. Un utilisateur a posé la question suivante en français: "{user_query}"
            L'API a retourné la réponse suivante:
            {api_response}
            Analyse cette réponse et fournis une réponse claire et concise en français qui répond directement à la question de l'utilisateur.
            Si l'API a retourné des données dans le champ "result" ou "answer", utilise ces informations pour formuler ta réponse.
            Si l'API a fourni une explication dans le champ "explanation" ou "references"(source de l'information), incorpore-la dans ta réponse.
            Si la question est une salutation (par exemple : "bonjour", "salut", etc.), répondre "Bonjour, comment puis-je vous aider ?" traduire {api_response} en français.
            Sinon répondre : « Je suis désolé, je n'ai pas compris votre question. Pourriez-vous la reformuler, s'il vous plaît ? ».
            Ne pas afficher l'UID de l'influenceur.
            Sois conversationnel et utile dans ton ton."""
            
        else:

            prompt = f"""
            You are a helpful virtual assistant designed to assist users with their inquiries. A user has asked the following question in English: "{user_query}". 
            Your task is to understand the {user_query}:
                - If the question is a greeting (e.g., 'hello', 'hi', etc.), respond with 'Hello, how can I help you?'. No need to include any other information.
                - Use the following API response to answer:{api_response} . Always show the result well structured and reformulate it — don't add any questions. 
                - Do not show the uid of the influencer.
                - Ensure your tone is conversational and helpful."""

        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return (
            f"Désolé, erreur lors de la génération : {e}" if language == "French"
            else f"Sorry, error generating response: {e}"
        )

# Main app
def main():
    st.title("🤖 Bilingual Chatbot Agent")
    st.markdown("*Ask me questions in French or English.*")

    if not st.session_state.api_configured and not configure_gemini():
        st.stop()

    with st.sidebar:
        st.header("ℹ️ About")
        st.write("This chatbot answers influencer and general data questions in English or French.")
        st.write("• English")
        st.write("• Français")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me a question in French or English..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                language = detect_language(prompt)
                api_response = call_api(prompt, language)

                if "answer" in api_response or "result" in api_response:
                    bot_response = generate_natural_response(api_response, prompt, language)
                else:
                    msg = (
                        api_response.get("error")
                        or api_response.get("message")
                        or json.dumps(api_response, indent=2)
                        or "Unknown error"
                    )
                    bot_response = (
                        f"Désolé, un problème est survenu : {msg}" if language == "French"
                        else f"Sorry, an issue occurred: {msg}"
                    )

                st.markdown(bot_response)

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

if __name__ == "__main__":
    main()
