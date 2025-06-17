# Bagian 1: Mengimpor "peralatan" yang dibutuhkan
import streamlit as st
import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# ---- MODIFIKASI: Kita ganti TextLoader dengan PyPDFLoader ----
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Memuat kunci rahasia dari file .env
load_dotenv()

# Bagian 2: Fungsi Persiapan AI dengan Cache
@st.cache_resource
def setup_ai_chain():
    """Fungsi untuk memuat SEMUA PDF dari sebuah folder, memprosesnya, dan menyiapkan rantai AI."""
    st.info("Memulai persiapan AI (hanya dijalankan saat pertama kali)...", icon="⏳")

    # Siapkan model AI Google
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    
    # ---- BAGIAN YANG DIMODIFIKASI SECARA TOTAL ----
    # Nama folder tempat menyimpan semua file PDF
    nama_folder = "sumber_pdf"
    all_docs = [] # List kosong untuk menampung semua teks dari semua PDF

    try:
        # Loop melalui setiap file di dalam folder yang ditentukan
        for filename in os.listdir(nama_folder):
            # Cek apakah file tersebut adalah file PDF
            if filename.lower().endswith('.pdf'):
                path_file = os.path.join(nama_folder, filename)
                st.write(f"Membaca file: {filename}...") # Info file yang sedang diproses
                loader = PyPDFLoader(path_file)
                # Tambahkan isi PDF ke dalam list utama
                all_docs.extend(loader.load())

        # Cek apakah ada dokumen yang berhasil dimuat
        if not all_docs:
            st.error(f"Tidak ada file PDF yang ditemukan di dalam folder '{nama_folder}'. Pastikan folder tidak kosong.")
            return None

    except FileNotFoundError:
        st.error(f"Folder '{nama_folder}' tidak ditemukan. Harap buat folder tersebut dan isi dengan file PDF.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")
        return None
    # ---- AKHIR DARI BAGIAN YANG DIMODIFIKASI ----

    # Pecah SEMUA dokumen yang terkumpul menjadi potongan-potongan kecil
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_docs)

    # Buat "lemari arsip" (vector store) dari semua potongan
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # Buat template perintah untuk AI
    prompt = ChatPromptTemplate.from_template("""
    Anda adalah asisten yang ahli dalam menganalisis dokumen. Jawab pertanyaan pengguna HANYA berdasarkan informasi di bawah ini.
    Sebutkan dari dokumen mana (jika informasi tersedia) jawaban Anda berasal.
    Jika jawaban tidak ada di informasi ini, katakan 'Maaf, saya tidak menemukan informasi tersebut di dalam dokumen yang tersedia.'

    Konteks:
    {context}

    Pertanyaan: {input}
    """)

    # Rangkai semua alat menjadi satu rantai proses
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    st.success("Persiapan AI selesai! Aplikasi siap digunakan.", icon="✅")
    return retrieval_chain

# --- Tampilan Website Sederhana Kita ---

# Bagian 3: Tampilan Antarmuka (User Interface)
st.title("🔎 Mesin Pencari Multi-Dokumen PDF")
st.write("Ajukan pertanyaan tentang isi dari semua PDF di folder 'sumber_pdf', dan AI akan mencarikan jawabannya.")

# Panggil fungsi persiapan AI
chain = setup_ai_chain()

# Pastikan chain berhasil dibuat sebelum melanjutkan
if chain:
    # Buat kotak input untuk pertanyaan
    pertanyaan_user = st.text_input("Ketik pertanyaan Anda di sini:")

    # Bagian 4: Logika Inti
    if pertanyaan_user:
        with st.spinner("Mencari jawaban di semua dokumen..."):
            response = chain.invoke({"input": pertanyaan_user})
            st.header("Jawaban:")
            st.write(response["answer"])

            # Tampilkan juga sumber informasinya (sekarang bisa berasal dari PDF yang berbeda)
            with st.expander("Lihat sumber informasi yang digunakan"):
                for doc in response["context"]:
                    sumber = doc.metadata.get('source', 'Tidak diketahui')
                    st.info(f"**Sumber:** `{sumber}`\n\n**Isi:**\n{doc.page_content}")