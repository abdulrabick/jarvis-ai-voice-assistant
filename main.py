import os
import re
import json
import hashlib

import chromadb
import ollama
import sounddevice as sd
import pyttsx3

from scipy.io.wavfile import write
from faster_whisper import WhisperModel
from pypdf import PdfReader


# ==========================================
# CONFIGURATION
# ==========================================

SAMPLE_RATE = 16000
RECORD_SECONDS = 5

MODEL_NAME = "qwen2.5:3b"
EMBEDDING_MODEL = "nomic-embed-text"

MIN_SIMILARITY = 0.65
TOP_K = 2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KNOWLEDGE_FILE = os.path.join(
    BASE_DIR, "my_knowledge.txt"
)

DOCUMENTS_FOLDER = os.path.join(
    BASE_DIR, "documents"
)

MEMORY_FILE = os.path.join(
    BASE_DIR, "memory.json"
)

AUDIO_FILE = os.path.join(
    BASE_DIR, "voice.wav"
)

CHROMA_FOLDER = os.path.join(
    BASE_DIR, "chroma_db"
)

COLLECTION_NAME = "python_knowledge"


# ==========================================
# TEXT-TO-SPEECH
# ==========================================

speaker = pyttsx3.init()

speaker.setProperty("rate", 165)
speaker.setProperty("volume", 1.0)


def speak(text):
    """Print and speak a message."""

    print(f"\nJARVIS: {text}")

    speaker.say(text)
    speaker.runAndWait()


# ==========================================
# CONVERSATION HISTORY
# ==========================================

conversation = []


# ==========================================
# VOICE RECOGNITION
# ==========================================

def listen(whisper_model):
    """Record audio and transcribe it using Whisper."""

    input("\nPress Enter to start recording...")

    print(
        f"Recording for {RECORD_SECONDS} seconds. Speak now!"
    )

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )

    sd.wait()

    write(
        AUDIO_FILE,
        SAMPLE_RATE,
        audio,
    )

    print("Converting speech to text...")

    segments, _ = whisper_model.transcribe(
        AUDIO_FILE,
        language="en",
        beam_size=5,
        vad_filter=True,
        initial_prompt=(
            "This is a Python programming lesson. "
            "Technical terms include Python, variable, "
            "if statement, for loop, while loop, function, "
            "list, dictionary, class, object, parameter, "
            "argument, Generative AI, and RAG."
        ),
        condition_on_previous_text=False,
    )

    recognized_text = " ".join(
        segment.text for segment in segments
    ).strip()

    print("You:", recognized_text)

    return recognized_text


# ==========================================
# PERSONAL KNOWLEDGE
# ==========================================

def load_knowledge():
    """Read personal knowledge from my_knowledge.txt."""

    if not os.path.exists(KNOWLEDGE_FILE):
        return ""

    with open(
        KNOWLEDGE_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        return file.read()


# ==========================================
# DOCUMENT LOADING: TXT AND PDF
# ==========================================

def load_documents():
    """
    Load TXT files and extract text from PDFs.

    Preserve each document's filename and PDF page number.
    """

    if not os.path.isdir(DOCUMENTS_FOLDER):
        print(
            f"Warning: Documents folder not found: "
            f"{DOCUMENTS_FOLDER}"
        )
        return []

    loaded_documents = []

    for filename in sorted(os.listdir(DOCUMENTS_FOLDER)):

        file_path = os.path.join(
            DOCUMENTS_FOLDER,
            filename,
        )

        lower_name = filename.lower()

        try:

            # Read TXT files.
            if lower_name.endswith(".txt"):

                with open(
                    file_path,
                    "r",
                    encoding="utf-8",
                ) as file:
                    content = file.read().strip()

                if content:
                    loaded_documents.append({
                        "source": filename,
                        "page": None,
                        "text": content,
                    })

                    print(
                        f"Loaded document: {filename}"
                    )

            # Read PDF files.
            elif lower_name.endswith(".pdf"):

                reader = PdfReader(file_path)

                previous_count = len(
                    loaded_documents
                )

                for page_number, page in enumerate(
                    reader.pages,
                    start=1,
                ):

                    page_text = (
                        page.extract_text() or ""
                    ).strip()

                    # Remove the sample test-question
                    # section from our practice PDF.
                    if lower_name == "python_guide_for_jarvis.pdf":

                        marker = (
                            "Suggested JARVIS test questions"
                        )

                        marker_position = (
                            page_text.lower().find(
                                marker.lower()
                            )
                        )

                        if marker_position != -1:
                            page_text = page_text[
                                :marker_position
                            ].strip()

                    # Save the extracted text with its source.
                    if page_text:
                        loaded_documents.append({
                            "source": filename,
                            "page": page_number,
                            "text": page_text,
                        })

                if len(loaded_documents) > previous_count:

                    print(
                        f"Loaded document: {filename}"
                    )

                else:

                    print(
                        f"Warning: No readable text found "
                        f"in {filename}. "
                        "If this is a scanned PDF, OCR is needed."
                    )

            # Ignore unsupported file formats.
            else:
                continue

        except Exception as error:

            print(
                f"Could not read {filename}: {error}"
            )

    return loaded_documents


# ==========================================
# DOCUMENT CHUNKING
# ==========================================

def split_into_chunks(documents, chunk_size=200):
    """
    Split documents into smaller chunks.

    Preserve the original filename and page number.
    """

    chunks = []

    for document in documents:

        text = re.sub(
            r"\s+",
            " ",
            document["text"],
        ).strip()

        if not text:
            continue

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        current_chunk = ""

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            if (
                current_chunk
                and len(current_chunk) + len(sentence) + 1
                > chunk_size
            ):

                chunks.append({
                    "source": document["source"],
                    "page": document.get("page"),
                    "text": current_chunk,
                })

                current_chunk = sentence

            else:

                current_chunk = (
                    f"{current_chunk} {sentence}"
                ).strip()

        if current_chunk:

            chunks.append({
                "source": document["source"],
                "page": document.get("page"),
                "text": current_chunk,
            })

    return chunks


# ==========================================
# EMBEDDINGS
# ==========================================

def get_embedding(text):
    """Convert text into a vector using Ollama."""

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=text,
    )

    return response["embeddings"][0]


# ==========================================
# PERSISTENT CHROMADB INDEX
# ==========================================

def build_document_index(chunks):
    """
    Store chunks and embeddings in ChromaDB.

    Reuse unchanged chunks on subsequent runs.
    """

    client = chromadb.PersistentClient(
        path=CHROMA_FOLDER
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        },
    )

    # Create stable IDs for document chunks.
    chunk_map = {}

    for chunk in chunks:

        identity = (
            f"{chunk['source']}\0"
            f"{chunk.get('page')}\0"
            f"{chunk['text']}"
        )

        chunk_id = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

        chunk_map[chunk_id] = chunk

    desired_ids = set(chunk_map.keys())

    # Retrieve existing database IDs.
    existing_records = collection.get(
        include=["metadatas"]
    )

    existing_ids = set(
        existing_records["ids"]
    )

    # Delete old chunks that are no longer in the documents.
    stale_ids = list(
        existing_ids - desired_ids
    )

    if stale_ids:

        collection.delete(
            ids=stale_ids
        )

        print(
            f"Removed {len(stale_ids)} "
            "outdated document chunks."
        )

    # Create embeddings only for new chunks.
    missing_ids = sorted(
        desired_ids - existing_ids
    )

    if missing_ids:

        print(
            f"\nCreating embeddings for "
            f"{len(missing_ids)} new chunks..."
        )

        batch_size = 10

        for start in range(
            0,
            len(missing_ids),
            batch_size,
        ):

            batch_ids = missing_ids[
                start:start + batch_size
            ]

            batch_chunks = [
                chunk_map[item_id]
                for item_id in batch_ids
            ]

            batch_texts = [
                item["text"]
                for item in batch_chunks
            ]

            batch_embeddings = [
                get_embedding(text)
                for text in batch_texts
            ]

            batch_metadatas = []

            for item in batch_chunks:

                metadata = {
                    "source": item["source"]
                }

                if item.get("page") is not None:

                    metadata["page"] = int(
                        item["page"]
                    )

                batch_metadatas.append(metadata)

            collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
            )

            saved_count = min(
                start + batch_size,
                len(missing_ids),
            )

            print(
                f"Saved {saved_count}/"
                f"{len(missing_ids)} new embeddings."
            )

    else:

        print(
            "\nAll document embeddings already exist. "
            "Reusing saved vectors."
        )

    print(
        f"Vector database ready: "
        f"{collection.count()} document chunks."
    )

    return collection


# ==========================================
# SEMANTIC SEARCH
# ==========================================

def retrieve_relevant_chunks(
    question,
    collection,
    top_k=TOP_K,
):
    """
    Search ChromaDB by semantic similarity.

    Return relevant chunks with their source information.
    """

    total_chunks = collection.count()

    if total_chunks == 0:

        print(
            "No documents are available for searching."
        )

        return []

    # Convert the question into an embedding.
    question_embedding = get_embedding(
        question
    )

    # Search the persistent vector database.
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, total_chunks),
        include=[
            "documents",
            "distances",
            "metadatas",
        ],
    )

    documents = (
        results.get("documents") or [[]]
    )[0]

    distances = (
        results.get("distances") or [[]]
    )[0]

    metadatas = (
        results.get("metadatas") or [[]]
    )[0]

    relevant_chunks = []

    print("\nSemantic search results:")

    for document, distance, metadata in zip(
        documents,
        distances,
        metadatas,
    ):

        similarity = 1.0 - float(distance)

        metadata = metadata or {}

        source = metadata.get(
            "source",
            "Unknown source",
        )

        page = metadata.get("page")

        page_label = (
            f", page {page}"
            if page is not None
            else ""
        )

        print(
            f"Similarity: {similarity:.4f}"
        )

        print(
            f"Source: {source}{page_label}"
        )

        print(
            f"Document: {document}\n"
        )

        # Keep only chunks above the threshold.
        if similarity >= MIN_SIMILARITY:

            source_label = (
                f"Source: {source}{page_label}"
            )

            relevant_chunks.append({
                "source": source,
                "page": page,
                "text": document,
                "similarity": similarity,
                "context": (
                    f"{source_label}\n{document}"
                ),
            })

    if not relevant_chunks:

        print(
            "No document chunks met the "
            "minimum similarity threshold."
        )

    return relevant_chunks


# ==========================================
# PERSONAL MEMORY
# ==========================================

def load_memory():
    """Load saved personal information from memory.json."""

    if not os.path.exists(MEMORY_FILE):
        return {}

    try:

        with open(
            MEMORY_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            memory = json.load(file)

        return memory if isinstance(memory, dict) else {}

    except (
        json.JSONDecodeError,
        OSError,
    ) as error:

        print(
            f"Warning: Could not read memory.json: {error}"
        )

        return {}


def save_memory(memory):
    """Write personal information to memory.json."""

    with open(
        MEMORY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            memory,
            file,
            indent=4,
            ensure_ascii=False,
        )


def update_memory(key, value):
    """Add or update a saved personal fact."""

    memory = load_memory()

    memory[key] = value

    save_memory(memory)


def handle_memory_command(user_text):
    """
    Handle commands such as:
    Remember that my favourite language is Python.
    """

    prefix = "remember that "

    if not user_text.lower().startswith(prefix):
        return None

    fact = user_text[len(prefix):].strip()

    match = re.match(
        r"(.+?)\s+is\s+(.+)",
        fact,
        flags=re.IGNORECASE,
    )

    if match:

        key = (
            match.group(1)
            .strip()
            .lower()
            .replace(" ", "_")
        )

        value = match.group(2).strip()

        if key and value:

            update_memory(
                key,
                value,
            )

            return (
                f"Okay, I saved that "
                f"{key.replace('_', ' ')} is {value}."
            )

    return (
        "Please use this format: "
        "Remember that my favourite language is Python."
    )


# ==========================================
# ASK QWEN
# ==========================================

def ask_qwen(user_text, vector_collection):
    """
    Answer using retrieved documents,
    personal knowledge, and saved memory.
    """

    knowledge = load_knowledge()

    memory = load_memory()

    relevant_chunks = retrieve_relevant_chunks(
        user_text,
        vector_collection,
        top_k=TOP_K,
    )

    if relevant_chunks:

        retrieved_text = "\n\n".join(
            item["context"]
            for item in relevant_chunks
        )

    else:

        retrieved_text = (
            "No document chunks passed "
            "the similarity threshold."
        )

    system_message = {
        "role": "system",
        "content": (
            "You are JARVIS, a helpful personal AI assistant. "
            "Answer clearly in simple language. "
            "Do not invent personal details.\n\n"

            f"Personal knowledge:\n"
            f"{knowledge}\n\n"

            f"Saved personal information:\n"
            f"{json.dumps(memory, indent=2)}\n\n"

            f"Retrieved document sections:\n"
            f"{retrieved_text}\n\n"

            "When your answer uses a retrieved section, "
            "identify its exact source filename. "
            "For a PDF, include the page number if provided. "
            "Never invent a source or page number.\n\n"

            "If no relevant document was retrieved, "
            "do not say the answer came from the user's files. "
            "You may answer general questions from your own "
            "knowledge, but distinguish that from file content.\n\n"

            "Do not claim to access live weather, news, "
            "or current information unless an appropriate "
            "external tool is connected.\n\n"

            "Answer the user's actual question. "
            "If the question seems garbled or incorrectly "
            "transcribed, ask the user to repeat it."
        ),
    }

    messages = [system_message]

    # Include recent conversation history.
    messages.extend(
        conversation[-10:]
    )

    messages.append({
        "role": "user",
        "content": user_text,
    })

    response = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
    )

    answer = response["message"]["content"]

    # Save this conversation in memory for the current session.
    conversation.append({
        "role": "user",
        "content": user_text,
    })

    conversation.append({
        "role": "assistant",
        "content": answer,
    })

    # Keep the conversation history limited.
    del conversation[:-10]

    return answer


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():

    print("=" * 45)

    print(
        "       JARVIS AI VOICE ASSISTANT"
    )

    print("=" * 45)

    speak(
        "Hello! I am Jarvis. "
        "Initializing my systems."
    )

    print("\nLoading Whisper AI...")

    whisper_model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8",
    )

    print(
        "Whisper loaded successfully!"
    )

    # Load TXT and PDF documents.
    documents = load_documents()

    # Split documents into chunks while retaining source information.
    chunks = split_into_chunks(
        documents
    )

    # Build or reuse the persistent vector index.
    vector_collection = build_document_index(
        chunks
    )

    speak(
        "All systems are ready. "
        "You can speak to me now."
    )

    while True:

        try:

            user_text = listen(
                whisper_model
            )

            if not user_text:

                speak(
                    "I couldn't understand that. "
                    "Please try again."
                )

                continue

            # Normalize punctuation in exit commands.
            command = (
                user_text
                .lower()
                .strip()
                .rstrip(" .!?\t\n")
            )

            if command in {
                 "exit",
                 "quit",
                 "bye",
                 "goodbye",
                 "stop jarvis",
                 }:
                
                speak("Goodbye! Shutting down.")
                break

                speak(
                    "Goodbye! Shutting down."
                )

                break

            # Handle personal memory commands.
            memory_answer = handle_memory_command(
                user_text
            )

            if memory_answer:

                speak(memory_answer)

                continue

            # Generate an answer using Qwen and retrieved documents.
            answer = ask_qwen(
                user_text,
                vector_collection,
            )

            speak(answer)

        except KeyboardInterrupt:

            speak("Goodbye!")

            break

        except Exception as error:

            print(
                "\nError:",
                error,
            )

            speak(
                "Sorry, something went wrong. "
                "Please check the terminal for details."
            )


if __name__ == "__main__":
    main()