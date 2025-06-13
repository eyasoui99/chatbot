from dotenv import load_dotenv
import streamlit as st
import requests
import json
import os
from langdetect import detect
import google.generativeai as genai
import pandas as pd
from datetime import datetime

# Load FAQ data
try:
    faq = pd.read_csv('faq_questions_answers.csv')
    # Ensure full column width and all rows are shown
    pd.set_option('display.max_colwidth', None)   # Show full text in each cell
    pd.set_option('display.max_rows', None)       # Show all rows
except FileNotFoundError:
    # Create empty DataFrame if CSV doesn't exist
    faq = pd.DataFrame(columns=['question', 'answer'])

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
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "api_configured" not in st.session_state:
    st.session_state.api_configured = False

# Gemini configuration
def configure_gemini():
    try:
        api_key = os.getenv("GEMINI_API_KEY", "")
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

# Add message to conversation history
def add_to_conversation_history(role, content, language="English"):
    """Add a message to the conversation history with metadata"""
    st.session_state.conversation_history.append({
        "role": role,
        "content": content,
        "language": language,
        "timestamp": datetime.now().isoformat()
    })
    
    # Keep only last 10 exchanges (20 messages) to manage memory
    if len(st.session_state.conversation_history) > 20:
        st.session_state.conversation_history = st.session_state.conversation_history[-20:]

# Check if current question relates to previous conversation
def is_question_related_to_context(current_query, conversation_history):
    """Use Gemini to determine if current question relates to previous conversation"""
    if not conversation_history or len(conversation_history) < 2:
        return False
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Get last few messages for context
        recent_history = conversation_history[-6:]  # Last 3 exchanges
        history_text = ""
        for msg in recent_history:
            history_text += f"{msg['role'].title()}: {msg['content']}\n"
        
        prompt = f"""
        Analyze if the current question relates to the previous conversation context.
        
        Recent conversation history:
        {history_text}
        
        Current question: "{current_query}"
        
        Determine if the current question:
        1. References something mentioned in the previous conversation (like "them", "it", "those", "the brands mentioned", "previously", etc.)
        2. Asks for more details about a previous topic
        3. Continues the same line of inquiry
        4. Uses pronouns or references that only make sense with the previous context
        
        Respond with only "YES" if the question is related to previous context, or "NO" if it's independent.
        """
        
        response = model.generate_content(prompt)
        result = response.text.strip().upper()
        return result == "YES"
    except Exception as e:
        print(f"Error checking question context: {e}")
        return False

# Reformulate query with context
def reformulate_query_with_context(current_query, conversation_history, language):
    """Reformulate the current query by incorporating relevant conversation context"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Get relevant conversation history
        recent_history = conversation_history[-6:]  # Last 3 exchanges
        history_text = ""
        for msg in recent_history:
            history_text += f"{msg['role'].title()}: {msg['content']}\n"
        
        if language == "French":
            prompt = f"""
            Tu dois reformuler une question en utilisant le contexte de la conversation précédente.
            
            Historique de conversation récent:
            {history_text}
            
            Question actuelle: "{current_query}"
            
            Reformule la question actuelle en une question complète et autonome qui incorpore les informations pertinentes de l'historique de conversation. La question reformulée doit être claire même sans le contexte de conversation.
            
            Ne retourne que la question reformulée, sans explication.
            """
        else:
            prompt = f"""
            You need to reformulate a question using the context from previous conversation.
            
            Recent conversation history:
            {history_text}
            
            Current question: "{current_query}"
            
            Reformulate the current question into a complete, standalone question that incorporates relevant information from the conversation history. The reformulated question should be clear even without the conversation context.
            
            Return only the reformulated question, no explanation.
            """
        
        response = model.generate_content(prompt)
        #print(response)
        return response.text.strip()
    except Exception as e:
        print(f"Error reformulating query: {e}")
        return current_query

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
        - Influencer personnel informations or details or profiles (e.g., influence themes, center of interest, email, country, my name, etc.)
        - Instagram community or follower insights
        - Statistics, audience, or Instagram performance details
        - Sales, clicks, or conversion rates related to a specific influencer, brand, or product
        - Information about brands or products
        - Lists or rankings of products/brands in specific categories (e.g., "top 10 products in X"), possibly with conditions (e.g., location-based filters)

        2. **analyze** → Use this label if the query is about legal documents, explanations, platform-related information, or general help. This includes:
        - If the query {query}, when translated to French, matches any item in {faq['question'].tolist() if not faq.empty else []}, it should be categorized as analyze.
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
        #print(f"Classification prompt: {prompt}")
        response = model.generate_content(prompt)
        label = response.text.strip().lower()
        #print(f"Classification result: {label}")
        return label if label in {"text2sql", "analyze", "web"} else "text2sql"
    except Exception as e:
        #print(f"Gemini classification failed: {e}")
        return "text2sql"

# Enhanced API handler with conversation context
def call_api(query, language, conversation_history, influencer_uid="la0NUVFtxnNnYng2JJF9i2FzkYz1"):
    """Enhanced API call that includes conversation context when relevant"""
    
    # Check if query relates to previous conversation
    is_contextual = is_question_related_to_context(query, conversation_history)
    
    # If contextual, reformulate the query
    if is_contextual:
        reformulated_query = reformulate_query_with_context(query, conversation_history, language)
        #print(f"Original query: {query}")
        #print(f"Reformulated query: {reformulated_query}")
        actual_query = reformulated_query
    else:
        actual_query = query
    
    query_type = classify_query(actual_query)

    if query_type == "web":
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                f"Tu es un assistant qui répond à des questions d'actualité en français. Réponds clairement à cette question sans poser de questions supplémentaires : {actual_query}"
                if language == "French"
                else f"You are a helpful assistant answering news/trend queries. Give a structured and concise answer without asking any follow-up questions: {actual_query}"
            )
            response = model.generate_content(prompt)
            return {"success": True, "result": response.text, "query_type": query_type, "was_reformulated": is_contextual}
        except Exception as e:
            return {"success": False, "error": f"Gemini error: {str(e)}", "query_type": query_type}

    # Prepare API call data
    url = (
        "https://text2sql-mffb.onrender.com/api/analyze"
        if query_type == "analyze"
        else "https://text2sql-mffb.onrender.com/api/query"
    )
    
    # Enhanced data structure with conversation context
    data = {
        "query": actual_query,
        "influencer_uid": influencer_uid
    }
    
    # Add conversation history if the question is contextual
    if is_contextual and conversation_history:
        # Format conversation history for API
        formatted_history = []
        for msg in conversation_history[-6:]:  # Last 3 exchanges
            if msg['role'] == 'user':
                formatted_history.append(f"Human: {msg['content']}")
            else:
                formatted_history.append(f"Assistant: {msg['content']}")
        
        data["conversation_context"] = {
            "has_context": True,
            "original_query": query,
            "reformulated_query": actual_query,
            "history": formatted_history
        }

    try:
        #print(f"API call to {url} with data: {json.dumps(data, indent=2)}")
        response = requests.post(url, headers={"Content-Type": "application/json"}, json=data, timeout=30)
        response.raise_for_status()
        api_response = response.json()
        api_response["query_type"] = query_type
        api_response["was_reformulated"] = is_contextual
        return api_response
    except requests.exceptions.Timeout:
        return {"success": False, "error": "API request timed out", "query_type": query_type}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"API error: {str(e)}", "query_type": query_type}
    except json.JSONDecodeError:
        return {"success": False, "error": "Invalid JSON returned from API", "query_type": query_type}

# Generate response using Gemini
def generate_natural_response(api_response, user_query, language):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Handle different response formats
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

        # Add context information if query was reformulated
        context_info = ""
        if api_response.get("was_reformulated", False):
            context_info = "\n(Note: This response considers the previous conversation context.)"

        if language == "French":
            prompt = f"""
            Tu es un assistant virtuel utile. Un utilisateur a posé la question suivante en français: "{user_query}"
            L'API a retourné la réponse suivante:
            {api_response}
            Analyse cette réponse et fournis une réponse claire et concise en français qui répond directement à la question de l'utilisateur.
            Si l'API a retourné des données dans le champ "result" ou "answer", utilise ces informations pour formuler ta réponse, et "result" doit être affiché entièrement. Pour que "result" soit clair, il doit être affiché sous forme de tableau.
            Si l'API a fourni une explication dans le champ "explanation" ou "references"(source de l'information), incorpore-la dans ta réponse.
            Si la question est une salutation (par exemple : "bonjour", "salut", etc.), répondre "Bonjour, comment puis-je vous aider ?" traduire {api_response} en français.
            Sinon répondre : « Je suis désolé, je n'ai pas compris votre question. Pourriez-vous la reformuler, s'il vous plaît ? ».
            Ne pas afficher l'UID de l'influenceur. 
            Ne pas indiquer que l'API a retourné ou d'après le résultat de l'API.
            Éviter les détails excessifs tels que « les informations sont extraites de la table X ».
            Sois conversationnel et utile dans ton ton.{context_info}"""
            
        else:
            prompt = f"""
                You are a helpful virtual assistant. A user asked the following question in English: "{user_query}"
                The API returned the following response:
                {api_response}
                Analyze this response and provide a clear and concise answer in English that directly addresses the user's question.
                If the API returned data in the "result" or "answer" field, use that information to formulate your response, and the content of "result" must be displayed in full. If the result needs to be clear, it should be displayed as a table.
                If the API provided an explanation in the "explanation" or "references" fields (sources of the information), incorporate it into your response.
                If the question is a greeting (e.g., "hello", "hi", etc.), respond with: "Hello, how can I help you?".
                Otherwise, respond: "I'm sorry, I didn't understand your question. Could you please rephrase it?"
                Do not display the influencer's UID.
                Avoid excessive details such as 'the information is extracted from table X'.
                Do not mention that the information came from the API or say "according to the API result."
                Ensure your tone is conversational and helpful.{context_info}"""

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
    st.markdown("*Ask me questions in French or English. I can understand context from our conversation.*")

    if not st.session_state.api_configured and not configure_gemini():
        st.stop()

    with st.sidebar:
        st.header("ℹ️ About")
        st.write("This chatbot answers influencer and general data questions in English or French.")
        st.write("**Features:**")
        st.write("• Multilingual support (English/Français)")
        st.write("• Conversation memory")
        st.write("• Context-aware responses")
        
        st.header("💬 Conversation")
        st.write(f"Messages in history: {len(st.session_state.conversation_history)}")
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.rerun()
        
        if st.button("New Topic"):
            st.session_state.conversation_history = []
            st.info("Conversation context cleared. Starting fresh topic.")
            st.rerun()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me a question in French or English..."):
        # Add user message to chat
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                language = detect_language(prompt)
                
                # Add user message to conversation history
                add_to_conversation_history("user", prompt, language)
                
                # Call API with conversation context
                api_response = call_api(prompt, language, st.session_state.conversation_history)

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
                
                # Add assistant response to conversation history
                add_to_conversation_history("assistant", bot_response, language)

        # Add messages to display history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

if __name__ == "__main__":
    main()
