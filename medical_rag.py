

print("1️⃣ Importing libraries...")
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
print("2️⃣ Libraries imported successfully!")

print("3️⃣ Importing Groq and FAISS...")
from groq import Groq
import faiss
print("4️⃣ Groq + FAISS imported successfully!")

print("5️⃣ Importing sentence-transformers...")
from sentence_transformers import SentenceTransformer
print("6️⃣ SentenceTransformer imported successfully!")

print("7️⃣ Loading .env...")
load_dotenv()
print("8️⃣ .env loaded!")

print("✅ Reached class definition!")

# -----------------------------
# Intent detection for non-medical queries
# -----------------------------
def detect_intent(query: str) -> str:
    text = query.lower().strip()

    greetings = ["hello", "hi", "hey", "good morning", "good evening", "good afternoon"]
    if any(greet in text for greet in greetings):
        return "greeting"

    appointment = ["appointment", "book", "schedule", "doctor visit", "consultation"]
    if any(word in text for word in appointment):
        return "appointment"

    thanks = ["thanks", "thank you", "bye", "goodbye"]
    if any(word in text for word in thanks):
        return "goodbye"

    return "medical"



class MedicalRAG:
    def __init__(self, medquad_csv="medquad.csv", disease_csv="disease_symptoms.csv"):
        print("🏥 Initializing Medical RAG System with Groq...")
        print("=" * 70)

        # ✅ Store file paths
        self.medquad_csv = medquad_csv
        self.disease_csv = disease_csv

        # 🔍 Environment setup
        from dotenv import load_dotenv
        load_dotenv()

        print("🔍 Checking .env and CSV files...")
        print("📂 Current working directory:", os.getcwd())
        print("📄 Files in current directory:", os.listdir())

        # ✅ API key
        self.api_key = os.getenv("GROQ_API_KEY")
        print("🔑 GROQ_API_KEY =", "✅ Found" if self.api_key else "❌ Missing")

        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables!")

        # ✅ Initialize Groq client
        from groq import Groq
        self.groq_client = Groq(api_key=self.api_key)
        self.groq_model = "llama-3.3-70b-versatile"  # new stable model

        # ✅ Load embedding model
        print("📥 Loading embedding model...")
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # ✅ Load datasets
        self.medquad_df = None
        self.disease_df = None
        self.load_datasets()

        # ✅ Build FAISS index
        self.index = None
        self.documents = []
        self.metadata = []
        self.build_index()

        print("✅ System Ready!\n")

    # -----------------------------
    # Load all datasets
    # -----------------------------
    def load_datasets(self):
        print("\n📊 Loading datasets...")

        # Load MedQuAD
        if os.path.exists("medquad.csv"):
            try:
                self.medquad_df = pd.read_csv("medquad.csv")
                self.medquad_df.columns = self.medquad_df.columns.str.strip()
                print(f"  ✓ MedQuAD loaded: {len(self.medquad_df)} records")
            except Exception as e:
                print(f"  ⚠️ Error loading medquad.csv: {e}")
                self.medquad_df = pd.DataFrame()
        else:
            print("  ⚠️ medquad.csv not found")
            self.medquad_df = pd.DataFrame()

        # Merge disease + symptom CSVs
        frames = []
        for fname in [
            "disease_symptoms.csv",
            "symptom_Description.csv",
            "symptom_precaution.csv",
            "Symptom-severity.csv",
        ]:
            if os.path.exists(fname):
                try:
                    df = pd.read_csv(fname)
                    df.columns = df.columns.str.strip()
                    df["source_file"] = fname
                    frames.append(df)
                    print(f"  ✓ {fname} loaded: {len(df)} records")
                except Exception as e:
                    print(f"  ⚠️ Error loading {fname}: {e}")
            else:
                print(f"  ⚠️ {fname} not found")

        if frames:
            self.disease_df = pd.concat(frames, ignore_index=True)
            print(f"  ✅ Combined disease dataset: {len(self.disease_df)} total records")
        else:
            self.disease_df = pd.DataFrame()
            print("  ⚠️ No disease CSVs loaded")

    # -----------------------------
    def safe_str(self, val):
        return "" if pd.isna(val) else str(val).strip()

    # -----------------------------
    def build_index(self):
        print("\n🔨 Building vector search index...")

        # Add MedQuAD data
        if self.medquad_df is not None and len(self.medquad_df) > 0:
            q_cols = [c for c in self.medquad_df.columns if "question" in c.lower()]
            a_cols = [c for c in self.medquad_df.columns if "answer" in c.lower()]
            if q_cols and a_cols:
                q_col, a_col = q_cols[0], a_cols[0]
                print(f"  Using MedQuAD columns: {q_col}, {a_col}")
                for _, row in self.medquad_df.iterrows():
                    q = self.safe_str(row[q_col])
                    a = self.safe_str(row[a_col])
                    if q and a:
                        self.documents.append(q)
                        self.metadata.append(
                            {"type": "qa", "question": q, "answer": a}
                        )

        # Add disease data
        if self.disease_df is not None and len(self.disease_df) > 0:
            all_cols = [c.lower() for c in self.disease_df.columns]
            d_cols = [c for c in self.disease_df.columns if "disease" in c.lower()]
            s_cols = [c for c in self.disease_df.columns if "symptom" in c.lower()]
            t_cols = [
                c
                for c in self.disease_df.columns
                if any(x in c.lower() for x in ["treatment", "precaution", "cure", "medicine"])
            ]

            d_col = d_cols[0] if d_cols else None
            s_col = s_cols[0] if s_cols else None
            t_col = t_cols[0] if t_cols else None

            print(
                f"  Using Disease columns: {d_col or '❌ none'}, {s_col or '❌ none'}, {t_col or '❌ none'}"
            )

            for _, row in self.disease_df.iterrows():
                disease = self.safe_str(row[d_col]) if d_col else ""
                symptoms = self.safe_str(row[s_col]) if s_col else ""
                treatment = (
                    self.safe_str(row[t_col])
                    if t_col and t_col in row
                    else "Treatment information not available"
                )
                if disease:
                    text = f"{disease}. Symptoms: {symptoms}"
                    self.documents.append(text)
                    self.metadata.append(
                        {
                            "type": "disease",
                            "disease": disease,
                            "symptoms": symptoms,
                            "treatment": treatment,
                        }
                    )

        if not self.documents:
            raise ValueError("❌ No documents to index! Check your CSV files.")

        print(f"  ✓ Total documents indexed: {len(self.documents)}")
        print("  ⏳ Creating embeddings...")

        embeddings = self.embedder.encode(
            self.documents, show_progress_bar=True, batch_size=32, convert_to_numpy=True
        )
        embeddings = embeddings.astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        print("  ✅ Vector index built successfully!")

    # -----------------------------
    def search(self, query: str, top_k: int = 5):
        if not query.strip():
            return []
        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(q_emb, min(top_k, len(self.documents)))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                score = 1.0 / (1.0 + float(dist))
                results.append(
                    {"metadata": self.metadata[idx], "score": score, "distance": float(dist)}
                )
        return results

    # -----------------------------
    def format_context(self, results: list[dict]) -> str:
        if not results:
            return "No relevant information found."
        parts = []
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            if meta["type"] == "qa":
                parts.append(f"[Source {i}]\nQ: {meta['question']}\nA: {meta['answer']}\n")
            else:
                parts.append(
                    f"[Source {i}]\nDisease: {meta['disease']}\nSymptoms: {meta['symptoms']}\nTreatment: {meta['treatment']}\n"
                )
        return "\n".join(parts)

    # -----------------------------
    def call_groq(self, system_prompt: str, user_prompt: str) -> str:
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.groq_model,
                temperature=0.1,
                max_tokens=1024,
                top_p=1,
                stream=False,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"❌ Groq API Error: {str(e)}"

    # -----------------------------
    def diagnose(self, query: str, show_sources: bool = False):
        # 🔍 Step 1: Detect intent first
        intent = detect_intent(query)

        # 🔹 Handle greetings and general conversation
        if intent == "greeting":
            return {
                "success": True,
                "response": "👋 Hello! I'm your AI medical assistant. How can I help you today?",
            }

        elif intent == "appointment":
            return {
                "success": True,
                "response": "🩺 To book a doctor’s appointment, please visit your nearest hospital’s website or call their patient helpline.",
            }

        elif intent == "goodbye":
            return {
                "success": True,
                "response": "😊 Take care! Remember to consult a healthcare provider for any medical concerns.",
            }

        # 🔹 Otherwise, process as a medical question (RAG pipeline)
        results = self.search(query, top_k=5)
        if not results:
            return {
                "success": False,
                "response": "No relevant information found. Please consult a healthcare provider.",
            }

        context = self.format_context(results)

        system_prompt = """You are a helpful medical assistant providing educational information based on the given knowledge base.
    Rules:
    1. Use ONLY provided info.
    2. Avoid specific medication dosages.
    3. Always recommend consulting a doctor.
    4. Emphasize safety.
    Format output clearly."""

        user_prompt = f"""Medical Knowledge Base:
    {context}

    Patient Query: {query}

    Generate a concise, clear educational response based ONLY on this context."""

        response_text = self.call_groq(system_prompt, user_prompt)

        response_data = {
            "success": True,
            "response": response_text,
        }

        if show_sources:
            response_data["sources"] = results

        return response_data



def main():
    """Run MedicalRAG standalone for debugging"""
    try:
        print("🚀 Starting MedicalRAG initialization...")
        rag = MedicalRAG()
        print("✅ MedicalRAG initialized successfully!")
        
        while True:
            q = input("💬 You: ").strip()
            if q.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye! Stay healthy.")
                break

            intent = detect_intent(q)

            if intent == "greeting":
                print("🤖 Bot: Hello! I’m your medical assistant. How can I help you today?\n")

            elif intent == "appointment":
                print("🤖 Bot: 🩺 To book an appointment, please visit your hospital’s website or call their helpline.\n")

            elif intent == "goodbye":
                print("🤖 Bot: 😊 Take care! Remember to stay hydrated and consult your doctor when needed.\n")

            else:
                res = rag.diagnose(q)
                print("\n" + "=" * 70)
                print("🤖 Bot:", res["response"])
                print("=" * 70 + "\n")


    except Exception as e:
        print("\n❌ CRASHED DURING STARTUP!")
        print(e)
        traceback.print_exc()



if __name__ == "__main__":
    main()
