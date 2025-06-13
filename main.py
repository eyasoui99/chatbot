from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import pandas as pd
import google.generativeai as genai
import requests
import json
from langdetect import detect

# Load .env
load_dotenv()

# Load FAQ
faq = pd.read_csv('faq_questions_answers.csv')

# FastAPI init
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Request model
class ChatRequest(BaseModel):
    query: str
    uid: str = "la0NUVFtxnNnYng2JJF9i2FzkYz1"

# Detect language
def detect_language(text):
    try:
        return "French" if detect(text) == "fr" else "English"
    except:
        return "English"

# Classify query
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
        return label if label in {"text2sql", "analyze", "web"} else "text2sql"
    except:
        return "text2sql"

# Call appropriate API or Gemini
def call_api(query, language, uid):
    query_type = classify_query(query)
    if query_type == "web":
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                f"Tu es un assistant qui répond à des questions d’actualité en français : {query}"
                if language == "French"
                else f"You are a helpful assistant answering queries: {query}"
            )
            response = model.generate_content(prompt)
            return {"success": True, "result": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}

    url = (
        "https://text2sql-mffb.onrender.com/api/analyze"
        if query_type == "analyze"
        else "https://text2sql-mffb.onrender.com/api/query"
    )
    data = {"query": query, "influencer_uid": uid}

    try:
        response = requests.post(url, json=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

# Generate natural language response
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
            Si l'API a retourné des données dans le champ "result" ou "answer", utilise ces informations pour formuler ta réponse, et "result" doit être affiché entiérement. Pour que "result" soit clair, il doit être affiché sous forme de tableau.
            Si l'API a fourni une explication dans le champ "explanation" ou "references"(source de l'information), incorpore-la dans ta réponse.
            Si la question est une salutation (par exemple : "bonjour", "salut", etc.), répondre "Bonjour, comment puis-je vous aider ?" traduire {api_response} en français.
            Sinon répondre : « Je suis désolé, je n'ai pas compris votre question. Pourriez-vous la reformuler, s'il vous plaît ? ».
            Ne pas afficher l'UID de l'influenceur. 
            Ne pas indiquer que l' API a retourné ou d'aprés le resultat de l'API.
            Sois conversationnel et utile dans ton ton."""
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
                Do not mention that the information came from the API or say "according to the API result."
                Ensure your tone is conversational and helpful."""

        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating response: {e}"

# API route
@app.post("/chatbot")
async def chatbot(request: ChatRequest):
    language = detect_language(request.query)
    api_response = call_api(request.query, language, request.uid)

    if "answer" in api_response or "result" in api_response:
        reply = generate_natural_response(api_response, request.query, language)
    else:
        msg = api_response.get("error") or api_response.get("message") or "Unknown error"
        reply = (
            f"Désolé, un problème est survenu : {msg}" if language == "French"
            else f"Sorry, an issue occurred: {msg}"
        )
    return {"response": reply}
