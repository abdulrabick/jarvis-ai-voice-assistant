# JARVIS AI Voice Assistant

A Python-based AI voice assistant that listens to spoken questions, searches personal documents, generates answers using a local large language model, and responds using text-to-speech.

The project combines voice recognition, Retrieval-Augmented Generation (RAG), semantic search, persistent vector storage, and personal memory.

## Features

* **Voice Recognition:** Converts spoken English into text using Faster-Whisper.
* **AI Responses:** Generates answers using the Qwen 2.5 3B model through Ollama.
* **Semantic Search:** Uses embeddings to retrieve relevant document sections based on meaning.
* **RAG Pipeline:** Supplies retrieved document context to the language model when answering questions.
* **PDF and TXT Support:** Extracts text from PDF and TXT files.
* **Persistent Vector Database:** Uses ChromaDB to store document chunks and embeddings.
* **Source Tracking:** Displays document filenames and PDF page numbers for retrieved sections.
* **Personal Memory:** Saves simple personal facts in a JSON file.
* **Text-to-Speech:** Speaks responses using pyttsx3.

## Technologies Used

* Python
* Ollama
* Qwen 2.5 3B
* Nomic Embed Text
* Faster-Whisper
* ChromaDB
* NumPy
* SoundDevice
* SciPy
* pyttsx3
* pypdf

## Project Structure

```text
JARVIS_AI/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── documents/
    ├── python_notes.txt
    └── Python_Guide_for_JARVIS.pdf
```

Additional files and folders are generated or created locally:

* `.venv/` — Python virtual environment.
* `chroma_db/` — Persistent vector database.
* `memory.json` — Saved personal facts.
* `my_knowledge.txt` — Personal knowledge file.
* `voice.wav` — Recorded audio.

These local files are excluded from Git using `.gitignore`.

## Prerequisites

Before running the application, install:

1. Python.
2. Ollama.
3. A working microphone.
4. The Python dependencies listed in `requirements.txt`.

The application requires an internet connection when downloading models for the first time. After the required models have been downloaded, inference runs locally through Ollama.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Replace `YOUR_USERNAME` and `YOUR_REPOSITORY` with your GitHub username and repository name.

Open the project directory:

```bash
cd YOUR_REPOSITORY
```

### 2. Create a virtual environment

On Windows, open PowerShell in the project directory and run:

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install the dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Download the Ollama models

Make sure Ollama is installed and running.

Download the language model:

```powershell
ollama pull qwen2.5:3b
```

Download the embedding model:

```powershell
ollama pull nomic-embed-text
```

The Qwen model generates answers. The embedding model converts document text and questions into numerical vectors for semantic search.

### 5. Prepare your personal knowledge file

Create a file named `my_knowledge.txt` in the project directory.

Example contents:

```text
My name is Abdul.
I am learning Python.
My goal is to become a Generative AI Engineer.
I am building a JARVIS AI voice assistant.
```

Replace these sample details with your own information.

### 6. Add documents

Place your `.txt` and text-based `.pdf` study materials inside the `documents` folder.

Example:

```text
documents/
├── python_notes.txt
└── Python_Guide_for_JARVIS.pdf
```

The application extracts text and indexes it for semantic search.

Scanned PDFs containing only images may require OCR before their contents can be searched.

## Running the Application

Activate your virtual environment, open the project directory in the terminal, and run:

```powershell
python main.py
```

Wait for Whisper and the document index to load.

When prompted:

1. Press Enter.
2. Speak your question in English.
3. Wait for the speech to be transcribed.
4. JARVIS searches the available documents.
5. Qwen generates a response.
6. JARVIS speaks the response.

Example questions:

* What is a while loop?
* What is the difference between a parameter and an argument?
* What are classes and objects in Python?
* How does exception handling work?

To exit, say "Goodbye", "Exit", or "Quit".

## How the RAG Pipeline Works

The application's document-question-answering process is:

1. Load PDF and TXT documents.
2. Split the extracted text into smaller chunks.
3. Generate document embeddings using `nomic-embed-text`.
4. Save the embeddings and source metadata in ChromaDB.
5. Convert the user's question into an embedding.
6. Retrieve the most similar document chunks.
7. Apply a similarity threshold to filter weaker matches.
8. Supply the retrieved context to Qwen.
9. Generate and speak the answer.

The application reuses stored embeddings for unchanged document chunks when it starts again.

## Personal Memory

JARVIS supports simple commands for saving personal facts.

Example:

```text
Remember that my favourite language is Python.
```

The fact is saved in `memory.json` and can remain available after restarting the application.

Do not store passwords, API keys, or other sensitive information in the memory file.

## Configuration

The main settings are defined near the top of `main.py`.

```python
SAMPLE_RATE = 16000
RECORD_SECONDS = 5

MODEL_NAME = "qwen2.5:3b"
EMBEDDING_MODEL = "nomic-embed-text"

MIN_SIMILARITY = 0.55
TOP_K = 2
```

* `RECORD_SECONDS` controls how long each audio recording lasts.
* `MODEL_NAME` selects the language model.
* `EMBEDDING_MODEL` selects the embedding model.
* `MIN_SIMILARITY` sets the minimum similarity score for retrieved chunks.
* `TOP_K` controls how many document chunks are retrieved.

These settings may need adjustment based on the documents and questions being tested.

## Troubleshooting

### Ollama connection error

Make sure Ollama is installed, running, and that both required models have been downloaded.

### Model not found

Run:

```powershell
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

### Microphone or audio error

Check your Windows microphone permissions and verify that the correct input device is available.

### Incorrect document retrieval

Check the semantic similarity scores and adjust `MIN_SIMILARITY` if necessary. A high similarity score does not guarantee that a retrieved chunk contains the correct answer.

### PDF text is missing

Confirm that the PDF contains extractable text. Image-only scanned documents may require OCR.

### Slow first startup

The application may need time to download or load Whisper models and generate embeddings for new documents.

## Limitations

* Voice input is configured for English transcription.
* Each recording lasts five seconds.
* PDF text extraction does not perform OCR on scanned documents.
* Personal memory supports simple key-value facts rather than unrestricted conversational memory.
* Answer quality depends on document quality, retrieval relevance, and the language model.
* A local Ollama model does not provide live weather, news, or other real-time information without an appropriate external tool.

## Future Improvements

* Add a graphical interface using Streamlit.
* Support additional document formats.
* Improve document chunking and retrieval quality.
* Add answer citations and stronger source-grounding checks.
* Add document management and deletion controls.
* Add automated tests and configurable application settings.

## Author

Abdul Rabick S

Aspiring Generative AI Engineer

## License

Choose a suitable open-source license before distributing the project. Until a license is added, do not assume that others have permission to reuse or redistribute the code.
