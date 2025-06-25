import os
import re
import gradio as gr
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key: 
    raise ValueError("OPENAI_API_KEY is missing from .env file")

# Function to extract YouTube transcript
def get_youtube_transcript(video_url):
    try:
        video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", video_url)
        if not video_id_match:
            return "Invalid YouTube URL. Please provide a valid link."
        
        video_id = video_id_match.group(1)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry["text"] for entry in transcript])
        return text
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

# Function to create vector embeddings from transcript
def create_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# Function to initialize chatbot memory
def initialize_chat(video_url):
    transcript = get_youtube_transcript(video_url)
    if "Error" in transcript:
        return transcript, None

    vector_store = create_vector_store(transcript)
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    return "Transcript loaded! You can start asking questions.", qa_chain

# Function to embed YouTube video
def embed_youtube_video(video_url):
    try:
        video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", video_url)
        if not video_id_match:
            return "Invalid YouTube URL."
        
        video_id = video_id_match.group(1)
        embed_html = f"""
        <iframe width="100%" height="315" src="https://www.youtube.com/embed/{video_id}" 
        frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; 
        gyroscope; picture-in-picture" allowfullscreen></iframe>
        """
        return embed_html
    except Exception:
        return "Could not embed the video. Please check the URL."

# Chat function
def chatbot_interface(chat_history, video_url, user_input):
    if not video_url:
        return chat_history + [("Error", "Please enter a YouTube video URL first.")]

    # Initialize chat only once
    if "qa_chain" not in chatbot_interface.__dict__:
        message, qa_chain = initialize_chat(video_url)
        chatbot_interface.qa_chain = qa_chain
        return chat_history + [(None, message)]

    # Process user question
    if chatbot_interface.qa_chain:
        answer = chatbot_interface.qa_chain.run(user_input)
        chat_history.append(("User", user_input))
        chat_history.append(("Bot", answer))
    else:
        chat_history.append(("Error", "Failed to process your request."))

    return chat_history

# Custom CSS for YouTube-style UI
custom_css = """
body {
    background-color: #0F0F0F;
    font-family: Arial, sans-serif;
}
.gradio-container {
    max-width: 550px;
    margin: auto;
    padding: 10px;
    text-align: center;
}
#header-container {
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative;
    background-color: #FF0000;
    padding: 5px 0;
    border-radius: 10px;
    margin-top: 20px; /* Added space above */
    margin-bottom: 20px;
}
#title {
    font-size: 42px;
    font-weight: bold;
    color: white;
    text-align: center;
    margin: 0;
}
#logo-container {
    text-align: center;
    margin-bottom: -30px; /* Moves the logo closer to the header */
}
#logo {
    width: 80px; /* Larger size */
    height: auto;
    margin: auto;
}
#chatbot {
    background-color: #FFFFFF;
    border-radius: 10px;
    padding: 5px;
    max-height: 250px;
    overflow-y: auto;
    font-size: 14px;
}
#video_input, #user_input {
    border: 2px solid #FF0000;
    border-radius: 5px;
    padding: 5px;
    font-size: 14px;
}
button {
    background-color: #FF0000 !important; /* Send button is now red */
    color: white !important;
    border-radius: 5px !important;
}
"""

# Gradio Chatbot UI
with gr.Blocks(css=custom_css) as app:
    with gr.Row(elem_id="logo-container"):
        gr.HTML('<img id="logo" src="https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg">')

    with gr.Row(elem_id="header-container"):
        gr.Markdown("🎤 **Ask AI About Any YouTube Video!**", elem_id="title")

    gr.Markdown("Enter a YouTube video URL to see the video and ask questions.")

    video_input = gr.Textbox(label="YouTube Video URL", elem_id="video_input", placeholder="Paste YouTube link here...")
    video_embed = gr.HTML()  # Placeholder for video
    chatbot = gr.Chatbot(label="Chat History", elem_id="chatbot")
    user_input = gr.Textbox(label="Your Question", elem_id="user_input", placeholder="Ask a question about the video...")

    send_button = gr.Button("Send")

    # Update the video embed when the URL is added
    video_input.change(embed_youtube_video, inputs=video_input, outputs=video_embed)

    send_button.click(
        chatbot_interface,
        inputs=[chatbot, video_input, user_input],
        outputs=chatbot,
    )

app.launch(debug=True)
