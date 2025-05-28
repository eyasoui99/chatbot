from dotenv import load_dotenv
import streamlit as st
import requests
import json
import os
from langdetect import detect
import google.generativeai as genai
load_dotenv()
# Configure the page
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

def configure_gemini():
    """Configure Gemini API"""
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            st.error("GEMINI_API_KEY environment variable not found. Please set your Gemini API key.")
            return False
        
        genai.configure(api_key=api_key)
        st.session_state.api_configured = True
        return True
    except Exception as e:
        st.error(f"Error configuring Gemini API: {str(e)}")
        return False

def detect_language(text):
    """Detect the language of the input text"""
    try:
        lang = detect(text)
        # Map language codes to full names
        if lang == "fr":
            return "French"
        elif lang == "en":
            return "English"
        else:
            return "English"  # Default to English
    except:
        return "English"  # Default to English if detection fails

def call_text2sql_api(query, influencer_uid="la0NUVFtxnNnYng2JJF9i2FzkYz1"):
    """Call the external text2sql API"""
    url = "https://text2sql-mffb.onrender.com/api/query"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "query": query,
        "influencer_uid": influencer_uid
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"success": False, "error": "API request timed out"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"API request failed: {str(e)}"}
    except json.JSONDecodeError:
        return {"success": False, "error": "Invalid JSON response from API"}

def generate_natural_response(api_response, user_query, language):
    """Generate natural language response"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prepare the prompt based on language
        if language == "French":
            prompt = f"""
            Tu es un assistant virtuel utile. Un utilisateur a posé la question suivante en français: "{user_query}"
            
            L'API a retourné la réponse suivante:
            {json.dumps(api_response, indent=2)}
            
            Analyse cette réponse et fournis une réponse claire et concise en français qui répond directement à la question de l'utilisateur.
            Si l'API a retourné des données dans le champ "result", utilise ces informations pour formuler ta réponse.
            Si l'API a fourni une explication dans le champ "explanation", incorpore-la dans ta réponse.
            Sois conversationnel et utile dans ton ton.
            """
        else:
            prompt = f"""
            You are a helpful virtual assistant. A user asked the following question in English: "{user_query}"
            
            The API returned the following response:
            {json.dumps(api_response, indent=2)}
            
            Analyze this response and provide a clear and concise answer in English that directly addresses the user's question.
            If the API returned data in the "result" field, use this information to formulate your response.
            If the API provided an explanation in the "explanation" field, incorporate it into your response.
            Be conversational and helpful in your tone.
            """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        if language == "French":
            return f"Désolé, j'ai rencontré une erreur lors de la génération de la réponse: {str(e)}"
        else:
            return f"Sorry, I encountered an error while generating the response: {str(e)}"

def process_user_query(user_input):
    """Process user query and generate response"""
    # Detect language
    language = detect_language(user_input)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Show loading spinner
    with st.spinner("Processing your query..."):
        # Call the text2sql API
        api_response = call_text2sql_api(user_input)
        
        # Process the API response
        if api_response.get("success", False):
            # Generate natural language response using Gemini
            bot_response = generate_natural_response(api_response, user_input, language)
        else:
            # Handle API errors
            error_msg = api_response.get("error", "Unknown error occurred")
            if language == "French":
                bot_response = f"Désolé, j'ai rencontré un problème lors du traitement de votre demande: {error_msg}"
            else:
                bot_response = f"Sorry, I encountered an issue while processing your request: {error_msg}"
    
    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

def main():
    """Main application function"""
    st.title("🤖 Bilingual Chatbot Agent")
    st.markdown("*Ask me questions in French or English about influencer data*")
    
    # Configure Gemini API
    if not st.session_state.api_configured:
        if not configure_gemini():
            st.stop()
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("This chatbot can answer questions about influencer data in both French and English.")
        st.write("It uses an external text2sql API and Gemini Flash 1.5 for natural language processing.")
        
        st.header("🌍 Supported Languages")
        st.write("• English")
        st.write("• Français")
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me a question in French or English..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Detect language
                language = detect_language(prompt)
                
                # Call the text2sql API
                api_response = call_text2sql_api(prompt)
                
                # Process the API response
                if api_response.get("success", False):
                    # Generate natural language response using Gemini
                    bot_response = generate_natural_response(api_response, prompt, language)
                else:
                    # Handle API errors
                    error_msg = api_response.get("error", "Unknown error occurred")
                    if language == "French":
                        bot_response = f"Désolé, j'ai rencontré un problème lors du traitement de votre demande: {error_msg}"
                    else:
                        bot_response = f"Sorry, I encountered an issue while processing your request: {error_msg}"
                
                st.markdown(bot_response)
        
        # Add messages to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

if __name__ == "__main__":
    main()
