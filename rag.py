import os
import pickle
from typing import List, Tuple
import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json

class PDFRAGSystem:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.ollama_url = "http://localhost:11434/api"
        
        self.chunks = []
        self.embeddings = None
        self.vector_store_path = "vector_store/embeddings.pkl"
    
    def get_embedding(self, text: str) -> List[float]:
        """Obtient les embeddings via Ollama avec mxbai-embed-large"""
        try:
            response = requests.post(
                f"{self.ollama_url}/embeddings",
                json={
                    "model": "mxbai-embed-large:latest",
                    "prompt": text
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["embedding"]
            else:
                raise Exception(f"Erreur Ollama: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Erreur d'embedding: {e}")
            # Fallback simple si Ollama n'est pas disponible
            return self.simple_embedding(text)
    
    def simple_embedding(self, text: str) -> List[float]:
        """Embedding de secours simple"""
        words = text.lower().split()
        unique_words = list(set(words))
        embedding = [words.count(word) for word in unique_words]
        # Normaliser
        if embedding:
            embedding = [x / max(embedding) for x in embedding]
        # Taille fixe
        fixed_size = 1024
        if len(embedding) < fixed_size:
            embedding.extend([0] * (fixed_size - len(embedding)))
        else:
            embedding = embedding[:fixed_size]
        return embedding
    
    def extract_text_from_pdf(self) -> str:
        """Extrait le texte du PDF"""
        text = ""
        try:
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            print(f"Erreur lors de l'extraction PDF: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
        """Découpe le texte en chunks"""
        # Méthode améliorée pour préserver la structure des phrases
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if sentence.strip():  # Ignorer les phrases vides
                if len(current_chunk) + len(sentence) < chunk_size:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
        
        # Ajouter le dernier chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def create_embeddings(self):
        """Crée les embeddings et les sauvegarde"""
        if os.path.exists(self.vector_store_path):
            print("Chargement des embeddings existants...")
            with open(self.vector_store_path, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.embeddings = np.array(data['embeddings'])
            return
        
        print("Création des embeddings avec mxbai-embed-large...")
        text = self.extract_text_from_pdf()
        if not text:
            raise ValueError("Aucun texte extrait du PDF")
            
        self.chunks = self.chunk_text(text)
        print(f"Nombre de chunks créés: {len(self.chunks)}")
        
        # Encodage avec mxbai-embed-large via Ollama
        print(f"Encodage de {len(self.chunks)} chunks...")
        self.embeddings = []
        
        for i, chunk in enumerate(self.chunks):
            if i % 5 == 0: 
                print(f"Embedding chunk {i+1}/{len(self.chunks)}")
            
            embedding = self.get_embedding(chunk)
            self.embeddings.append(embedding)
        
        self.embeddings = np.array(self.embeddings)
        
        # Sauvegarde
        os.makedirs("vector_store", exist_ok=True)
        with open(self.vector_store_path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'embeddings': self.embeddings.tolist()
            }, f)
        print("Embeddings sauvegardés avec succès!")
    
    def search_similar_chunks(self, query: str, top_k: int = 4) -> List[Tuple[str, float]]:
        """Recherche les chunks les plus similaires"""
        query_embedding = np.array(self.get_embedding(query)).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Trie par similarité
        sorted_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(self.chunks[i], similarities[i]) for i in sorted_indices if similarities[i] > 0.1]  # Seuil minimum
        
        return results if results else []
    
    def generate_answer(self, query: str, context: str) -> str:
        """Génère une réponse basée sur le contexte avec llama3.2 via Ollama"""
        prompt = f"""<|start_header_id|>system<|end_header_id|>

Vous êtes un assistant utile qui répond aux questions basées sur le contexte fourni. Suivez ces règles strictement:

1. Répondez UNIQUEMENT en utilisant les informations du contexte fourni
2. Si la réponse n'est pas dans le contexte, dites "Je n'ai pas trouvé cette information dans le document"
3. Soyez précis et concis
4. Répondez en français

CONTEXTE:
{context}

QUESTION: {query}

Répondez de manière claire en vous basant uniquement sur le contexte.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
        
        try:
            response = requests.post(
                f"{self.ollama_url}/generate",
                json={
                    "model": "llama3.2:latest",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 40,
                        "num_predict": 512,
                        "repeat_penalty": 1.1
                    }
                },
                timeout=120  # Timeout plus long pour la génération
            )
            
            if response.status_code == 200:
                full_response = response.json()["response"]
                # Nettoyer la réponse (enlever le prompt si présent)
                if prompt in full_response:
                    full_response = full_response.replace(prompt, "").strip()
                return full_response
            else:
                return f"Erreur lors de la génération (code {response.status_code}): {response.text}"
                
        except requests.exceptions.ConnectionError:
            return "❌ Erreur: Ollama n'est pas démarré. Veuillez lancer 'ollama serve' dans un terminal."
        except requests.exceptions.Timeout:
            return "⏰ Erreur: Timeout - La génération a pris trop de temps."
        except Exception as e:
            return f"❌ Erreur de génération: {str(e)}"
    
    def ask_question(self, question: str) -> str:
        """Pose une question et retourne une réponse"""
        try:
            print(f"Recherche de similarités pour: {question}")
            similar_chunks = self.search_similar_chunks(question)
            
            if not similar_chunks:
                return "🔍 Je n'ai pas trouvé d'informations pertinentes dans le document pour répondre à votre question."
            
            # Formatage du contexte
            context_parts = []
            for i, (chunk, score) in enumerate(similar_chunks):
                context_parts.append(f"[Source {i+1} - Pertinence: {score:.3f}]\n{chunk}")
            
            context = "\n\n".join(context_parts)
            
            print(f"Contexte récupéré ({len(similar_chunks)} chunks, scores: {[f'{score:.3f}' for _, score in similar_chunks]})")
            print("Génération de la réponse avec llama3.2...")
            
            answer = self.generate_answer(question, context)
            return answer
            
        except Exception as e:
            return f"❌ Erreur lors du traitement de la question: {str(e)}"

    def get_model_info(self) -> dict:
        """Retourne des informations sur les modèles utilisés"""
        return {
            "embedding_model": "mxbai-embed-large:latest (via Ollama)",
            "generation_model": "llama3.2:latest (via Ollama)",
            "embedding_dimension": self.embeddings.shape[1] if self.embeddings is not None else None,
            "chunks_count": len(self.chunks),
            "ollama_status": "✅ Connecté" if self.test_ollama_connection() else "❌ Non connecté"
        }
    
    def test_ollama_connection(self) -> bool:
        """Teste la connexion à Ollama"""
        try:
            response = requests.get(f"{self.ollama_url}/tags", timeout=5)
            return response.status_code == 200
        except:
            return False