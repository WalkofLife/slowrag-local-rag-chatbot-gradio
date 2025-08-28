# slowrag-local-rag-chatbot-gradio
A Retrieval-Augmented Generation (RAG) chatbot that runs entirely on CPU with a local mistral model. Interface built using Gradio. Slow, kinda dumb, but free, private, and 100% yours.

🦙 Laggy Llama — A Local RAG Chatbot on CPU

So here’s the deal:
This is a local Retrieval-Augmented Generation (RAG) chatbot. It runs on my laptop with no GPU.

Does it suck? Oh yeah.
Is it slow? Absolutely.
Does it still kinda work? You bet.

But it’s free, private, and mine. That’s the charm.

👉 Full write-up here: Medium Article

🚀 What is this?
* Upload a PDF.
* Ask a question.
* Watch as a poor CPU-based llama struggles to give you an answer.

Behind the scenes:
* Embeddings (BGE-small from HuggingFace) do the heavy lifting.
* ChromaDB stores PDF chunks.
* CTransformers runs a local LLaMA model (slowly, but surely).
* LangChain glues it all together.
* Gradio gives you a shiny web UI.

⚙️ Installation

Clone this repo:

git clone https://github.com/WalkofLife/slowrag-local-rag-chatbot-gradio
.git


Install dependencies:
pip install -r requirements.txt

📂 Files
* config.yaml → points to your local LLM model file.
* configure_llm.py → wrapper to load the model.
* customer_support_chatbot.py (yeah, misnamed — it’s actually just a general PDF Q&A bot).
* requirements.txt → dependencies list.

🛠️ Usage
1. Put your GGML/GGUF model file somewhere (e.g. models/llama-2-7b-chat.ggmlv3.q4_0.bin).
2. Update config.yaml with the correct path.
3. Run: python customer_support_chatbot.py
4. Open http://127.0.0.1:7860 in your browser.
5. Upload a PDF, ask a question, and… wait.

(Answers may vary between surprisingly good, hilariously bad, and “why are you talking about pineapples?”)

🐌 Limitations (aka The Fun Part)
* Runs entirely on CPU → slow as heck.
* Model quality → not GPT-4, more like GPT-0.5.
* Long answers? Grab coffee. Maybe two.
* Sometimes it hallucinates random nonsense.

🎯 Why Even Bother?
* No API bills.
* Full privacy — your data never leaves your laptop.
* Cool factor — you can say you built your own AI chatbot.
* Fun — because building janky AI is more satisfying than renting a perfect one.

Or, get a GPU (but then you lose the “laggy” charm).

🐻 License

MIT — do whatever you want, just don’t blame me when it gives you bad answers.
