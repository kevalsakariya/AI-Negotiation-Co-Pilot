import os
import base64
import streamlit as st
import google.generativeai as genai
import rag_processor  
import time


st.set_page_config(
    page_title="AI Negotiation Co-pilot",
    page_icon="🤝",
    layout="wide"
)


try:
    from dotenv import load_dotenv
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except ImportError:
    st.error("Please install `python-dotenv` via `pip install python-dotenv`.")
    st.stop()
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()

model = genai.GenerativeModel('models/gemini-2.0-flash')


# --- App State ---
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'temp_pdf_path' not in st.session_state:
    st.session_state.temp_pdf_path = ""

# --- UI Layout ---
st.title("AI-Powered Negotiation Co-pilot 🚀")
st.markdown("Upload a rules document (PDF) and a negotiation audio file, then ask questions.")

# --- Sidebar for File Uploads ---
with st.sidebar:
    st.header("1. Upload Your Documents")

    uploaded_pdf = st.file_uploader("Upload Rules PDF", type="pdf")
    if uploaded_pdf:
        temp_dir = "temp_files"
        os.makedirs(temp_dir, exist_ok=True)
        pdf_path = os.path.join(temp_dir, uploaded_pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        st.session_state.temp_pdf_path = pdf_path

        if st.button("Process Rules Document"):
            with st.spinner("Processing PDF..."):
                chunks, embeddings = rag_processor.process_pdf(st.session_state.temp_pdf_path)
                if chunks and embeddings is not None:
                    rag_processor.create_and_save_faiss_index(chunks, embeddings)
                    st.session_state.pdf_processed = True
                    st.success("✅ PDF processed and indexed successfully!")
                else:
                    st.error("Failed to process PDF.")

    st.divider()

    if st.session_state.pdf_processed:
        st.header("2. Upload Negotiation Audio")
        uploaded_audio = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a"])
    else:
        st.info("Please process a PDF document first.")
        uploaded_audio = None

# --- Main Chat Interface ---
if st.session_state.pdf_processed and uploaded_audio:
    st.header("3. Ask Your Questions")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("e.g., Ask about specific negotiation clauses..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):

                # --- Step 1: Save audio file locally ---
                temp_audio_path = os.path.join("temp_files", uploaded_audio.name)
                with open(temp_audio_path, "wb") as f:
                    f.write(uploaded_audio.read())

                # --- Step 2: Read audio and base64 encode it ---
                try:
                    with open(temp_audio_path, "rb") as f:
                        audio_bytes = f.read()
                    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

                    audio_part = {
                        "inline_data": {
                            "mime_type": uploaded_audio.type,  # e.g., audio/wav
                            "data": audio_b64
                        }
                    }
                except Exception as e:
                    st.error(f"Error reading/encoding audio: {e}")
                    st.stop()

                # --- Step 3: Transcribe via Gemini ---
                try:
                    st.write("Transcribing audio...")
                    transcription_response = model.generate_content(
                        ["Please transcribe this audio file.", audio_part],
                        request_options={'timeout': 600}
                    )
                    transcript = transcription_response.text
                    st.success("✅ Audio transcribed successfully.")
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    st.stop()

                # --- Step 4: Retrieve relevant rules from PDF ---
                st.write("Retrieving relevant rules from the PDF...")
                relevant_chunks = rag_processor.retrieve_relevant_chunks(prompt)
                context_rules = "\n".join(relevant_chunks)

                # --- Step 5: Generate the final answer ---
                st.write("Generating final answer...")
                final_prompt = f"""
                    You are an AI Negotiation Analyst. Your task is to answer the user's question based on the provided negotiation transcript and the relevant rule clauses from a document.

                    **Relevant Rule Clauses:**
                    ---
                    {context_rules if context_rules else "No specific rules found related to the question."}
                    ---

                    **Negotiation Transcript:**
                    ---
                    {transcript}
                    ---

                    **User's Question:** {prompt}

                    **Provide a clear, concise answer based *only* on the information in the transcript and the rules provided.** If the information isn't available, state that.
                    """

                try:
                    response = model.generate_content(final_prompt)
                    full_response = response.text
                    st.markdown(full_response)
                except Exception as e:
                    st.error(f"Response generation failed: {e}")
                    full_response = "Sorry, I couldn't generate a response."

                st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.info("Please upload and process a PDF, then upload an audio file to begin the analysis.")



