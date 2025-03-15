import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from langchain_community.tools import YouTubeSearchTool
from statsmodels.tsa.arima.model import ARIMA
import re
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.tools import YouTubeSearchTool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun




# ... [Keep previous CSS and initial setup code] ...

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }

    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }

    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        border-radius: 10px;
        margin: 10px 0;
    }

    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        border-radius: 10px;
        margin: 10px 0;
    }

    .video-card {
        transition: transform 0.2s;
        border-radius: 10px;
        overflow: hidden;
        margin: 10px 0;
    }

    .video-card:hover {
        transform: scale(1.05);
    }

    .wiki-box {
        background-color: #2A2A2A;
        border-left: 4px solid #00FFAA;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
    }

    .stMarkdown h3 {
        color: #00FFAA !important;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If the context is insufficient or you're unsure, state that you don't know. 
Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

PDF_STORAGE_PATH = 'document_store/'
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")
YOUTUBE_TOOL = YouTubeSearchTool()
WIKI_API = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
WIKI_TOOL = WikipediaQueryRun(api_wrapper=WIKI_API)


def extract_video_id(url):
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None


def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def load_pdf_documents(file_path):
    return PDFPlumberLoader(file_path).load()


def chunk_documents(raw_docs):
    return RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(raw_docs)


def index_documents(chunks):
    DOCUMENT_VECTOR_DB.add_documents(chunks)


def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)


# Stock Prediction Functions
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="60y")
        return hist
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return None


def predict_stock_price(ticker):
    data = get_stock_data(ticker)
    if data is None or data.empty:
        return None, None

    try:
        close_prices = data['Close']
        model = ARIMA(close_prices, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=5)

        plt.figure(figsize=(10, 5))
        plt.plot(close_prices, label='Historical Price')
        plt.plot(forecast, label='Forecast', color='red')
        plt.title(f'{ticker} Stock Price Prediction')
        plt.legend()
        return plt, forecast
    except Exception as e:
        st.error(f"ARIMA model error: {str(e)}")
        return None, None


# URL Sentiment Analysis
def analyze_url_content(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        return TextBlob(text).sentiment
    except Exception as e:
        st.error(f"Error analyzing URL: {str(e)}")
        return None


# URL Validation Function
def is_valid_url(text):
    url_pattern = re.compile(
        r'^(https?|ftp)://'  # Match http, https, or ftp
        r'(([A-Z0-9]([A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}|'  # Match domain name
        r'(\d{1,3}\.){3}\d{1,3}|'  # Match IP address
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # Match IPv6
        r'(:\d+)?'  # Optional port
        r'(/\S*)?$',  # Path
        re.IGNORECASE
    )
    return re.match(url_pattern, text) is not None
def fetch_youtube_links(query):
    tool = YouTubeSearchTool()
    response = tool.run(query)  # Fetch URLs as a string
    urls = response.strip().split(",")  # Convert to list (assuming comma-separated)

    # Clean URLs: Remove extra quotes or whitespace
    return [url.strip().strip("'").strip('"') for url in urls if url.strip()]

def generate_answer(query, context_docs):
    context = "\n\n".join([d.page_content for d in context_docs])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | LANGUAGE_MODEL
    return chain.invoke({"user_query": query, "document_context": context})


# Function to display video thumbnails in a grid (3 per row)
def display_video_thumbnails(youtube_urls, max_videos=9):
    if not youtube_urls:
        st.warning("No valid YouTube links found!")
        return

    cols = st.columns(3)  # Create 3 columns for layout

    for index, url in enumerate(youtube_urls[:max_videos]):  # Limit to max_videos (5)
        video_id = url.split("v=")[1].split("&")[0] if "watch?v=" in url else None
        if video_id:
            thumbnail_url = f"https://img.youtube.com/vi/{video_id}/0.jpg"
            clean_url = url.strip().replace("'", "").replace('"', "")
            with cols[index % 3]:  # Place thumbnails in a 3-column layout
                st.markdown(
                    f'<a href="{clean_url}" target="_blank">'
                    f'<img src="{thumbnail_url}" width="100%"></a>',
                    unsafe_allow_html=True
                )
        else:
            st.warning(f"Invalid YouTube URL: {url}")


# Modified Chat Handling
st.title("📘 DocuMind AI")
st.markdown("### Your Intelligent Research Assistant")
st.markdown("---")

# File Upload
uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Maximum file size: 50MB"
)

if uploaded_pdf:
    with st.spinner("Processing document..."):
        file_path = save_uploaded_file(uploaded_pdf)
        raw_docs = load_pdf_documents(file_path)
        chunks = chunk_documents(raw_docs)
        index_documents(chunks)
    st.success("Document processed successfully!")

user_input = st.chat_input("Ask your question or request resources...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    # URL Sentiment Analysis
    if is_valid_url(user_input):
        with st.chat_message("assistant", avatar="🤖"):
            st.subheader("🌐 URL Sentiment Analysis")
            sentiment = analyze_url_content(user_input)
            if sentiment:
                polarity = sentiment.polarity
                subjectivity = sentiment.subjectivity

                col1, col2 = st.columns(2)
                col1.metric("Sentiment Polarity", f"{polarity:.2f}")
                col2.metric("Subjectivity", f"{subjectivity:.2f}")

                if polarity > 0.2:
                    st.success("Strong Positive Sentiment")
                elif polarity > 0:
                    st.info("Positive Sentiment")
                elif polarity < -0.2:
                    st.error("Strong Negative Sentiment")
                else:
                    st.warning("Neutral Sentiment")
            else:
                st.error("Could not analyze content from this URL")

    # Stock Prediction
    elif user_input.isupper() and 1 <= len(user_input) <= 5:
        with st.chat_message("assistant", avatar="🤖"):
            try:
                st.subheader(f"📈 {user_input} Stock Analysis")
                data = get_stock_data(user_input)

                if data is None or data.empty:
                    st.error("Could not fetch stock data for this ticker")

                current_price = data['Close'].iloc[-1] if len(data['Close']) > 0 else None

                if current_price is None:
                    st.error("No valid price data available")

                with st.spinner("Generating predictions..."):
                    try:
                        close_prices = data['Close']
                        # Use iloc for positional indexing
                        model = ARIMA(close_prices, order=(5, 1, 0))
                        model_fit = model.fit()
                        forecast = model_fit.forecast(steps=5)

                        # Plotting
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(close_prices, label='Historical Price')
                        ax.plot(forecast, label='Forecast', color='red')
                        ax.set_title(f'{user_input} Stock Price Prediction')
                        ax.legend()
                        st.pyplot(fig)

                        # Forecast display
                        st.subheader("Next 5 Days Forecast")
                        st.line_chart(forecast)

                        # Calculate percentage change safely
                        if len(forecast) > 0 and current_price != 0:
                            change_percent = (forecast.iloc[-1] - current_price) / current_price * 100
                            st.metric("Predicted Change",
                                      f"{change_percent:.2f}%",
                                      delta_color="inverse")
                        else:
                            st.warning("Could not calculate price change")

                    except Exception as model_error:
                        st.error(f"Prediction failed: {str(model_error)}")

            except Exception as e:
                st.error(f"Stock analysis error: {str(e)}")

    # Existing Document/Youtube/Wikipedia Logic
    else:
        video_keywords = ['video', 'youtube', 'watch', 'demonstration', 'tutorial']
        is_video_request = any(kw in user_input.lower() for kw in video_keywords)

        if is_video_request:
            with st.chat_message("assistant", avatar="🤖"):
                st.subheader("🎥 Related Videos")
                try:
                    youtube_urls = fetch_youtube_links(f"{user_input},9")
                    display_video_thumbnails(youtube_urls, max_videos=10)
                except Exception as e:
                    st.error("Error fetching video results. Please try again.")
        else:
            with st.spinner("Analyzing documents..."):
                relevant_docs = find_related_documents(user_input)
                answer = generate_answer(user_input, relevant_docs)

            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(answer)

                # Add Wikipedia information
                try:
                    wiki_info = WIKI_TOOL.run(user_input)
                    if wiki_info and 'may refer to:' not in wiki_info:
                        st.markdown("---")
                        st.markdown("#### 📚 Wikipedia Insights")
                        st.markdown(f'<div class="wiki-box">{wiki_info}</div>',
                                    unsafe_allow_html=True)
                except Exception as e:
                    st.error("Error accessing Wikipedia information")

# Add stock prediction controls to sidebar
with st.sidebar:
    st.markdown("## 📈 Stock Analysis")
    st.markdown("""
    Enter stock tickers directly in chat (e.g., `AAPL` or `GOOGL`)
    - Get 1-year historical data
    - ARIMA price prediction
    - 5-day forecast visualization
    """)
    st.markdown("---")
    st.markdown("## 🌐 Web Analysis")
    st.markdown("""
    Paste any URL to get:
    - Content sentiment analysis
    - Polarity scoring
    - Subjectivity assessment
    """)

# ... [Keep remaining existing code] ...