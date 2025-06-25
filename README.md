# 📺 Ask AI About Any YouTube Video

This project lets you interact with any YouTube video using AI.  
Just paste a YouTube link, and the chatbot will extract the transcript and answer your questions based on the content of the video.

---

##  Features

-  Extracts transcript from any YouTube video  
-  Uses OpenAI GPT-4 to answer questions  
-  Splits and indexes video transcript using vector embeddings  
-  Clean user interface built with Gradio  
-  Environment variables support via `.env`

---

##  Tech Stack

- **Python**
- **Gradio**
- **LangChain**
- **FAISS**
- **OpenAI GPT-4**
- **YouTube Transcript API**

---

## 📦 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/sarahAljuhani2/Ask-AI-About-Any-YouTube-Video.git
cd Ask-AI-About-Any-YouTube-Video

# 2. Create virtual environment
python -m venv myenv
# Activate it:
# On Windows:
myenv\Scripts\activate
# On Mac/Linux:
source myenv/bin/activate

# 3. Install required packages
pip install -r requirements.txt

# 4. Add your OpenAI API key to a .env file
# Example content of .env:
OPENAI_API_KEY=your_openai_key_here

# 5. Run the app
python app.py
