import streamlit as st
import os
from rag import PDFRAGSystem
import time

# Configuration de la page
st.set_page_config(
    page_title="Chatbot PDF Local 🤖",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .model-info {
        background-color: #e8f4fd;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    .status-connected {
        color: #28a745;
        font-weight: bold;
    }
    .status-disconnected {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Chemin spécifique vers votre document PDF
PDF_PATH = "mon rapport.pdf" 

@st.cache_resource
def load_rag_system():
    """Charge le système RAG avec le chemin spécifique du document"""
    
    if not os.path.exists(PDF_PATH):
        st.error(f"""
        ❌ **Document PDF non trouvé !**
        
        Le fichier spécifié n'existe pas:
        `{PDF_PATH}`
        
        Veuillez:
        1. Vérifier que le fichier existe à cet emplacement
        2. Ou modifier la variable `PDF_PATH` dans app.py
        3. Assurez-vous que le document est bien un PDF valide
        """)
        return None
    
    try:
        with st.spinner("🔄 Initialisation du système RAG avec les modèles locaux..."):
            rag_system = PDFRAGSystem(PDF_PATH)
            
            # Tester la connexion Ollama
            if not rag_system.test_ollama_connection():
                st.error("""
                ❌ **Ollama n'est pas démarré !**
                
                Veuillez lancer Ollama avec la commande:
                ```bash
                ollama serve
                ```
                """)
                return None
            
            rag_system.create_embeddings()
            st.success(f"✅ Document chargé: `{os.path.basename(PDF_PATH)}`")
            return rag_system
            
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Chatbot PDF Local</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Powered by Ollama • mxbai-embed-large • llama3.2</p>', unsafe_allow_html=True)
    
    # Afficher le chemin du document
    st.info(f"**Document utilisé:** `{PDF_PATH}`")
    
    # Chargement du système
    rag_system = load_rag_system()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        if rag_system:
            info = rag_system.get_model_info()
            
            st.markdown("### 📊 Modèles Locaux")
            st.markdown(f"""
            <div class="model-info">
            <strong>Embedding:</strong> {info['embedding_model']}<br>
            <strong>Génération:</strong> {info['generation_model']}<br>
            <strong>Dimensions:</strong> {info['embedding_dimension']}<br>
            <strong>Segments:</strong> {info['chunks_count']}<br>
            <strong>Statut Ollama:</strong> <span class="status-connected">{info['ollama_status']}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.header("🎯 Contrôles")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Effacer Chat", use_container_width=True):
                if "messages" in st.session_state:
                    st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("🔄 Recharger", use_container_width=True):
                st.cache_resource.clear()
                st.rerun()
        
        st.markdown("---")
        st.header("❓ Exemples de Questions")
        st.markdown("""
        - *"Qu’est-ce que l’IA explicable (XAI) ?"*
        - *"Pourquoi l’IA explicable est-elle importante selon le rapport"*
        - *"Qu’est-ce que LIME, et comment ça fonctionne ?"*
        - *"Qu’est-ce que SHAP et d’où vient ce concept ?"*
        - *"Quel est le “trade-off” (compromis) entre performance du modèle et explicabilité ?"*
        - *"Comment la XAI peut contribuer à l’éthique et à la responsabilité dans les systèmes IA ?"*
        """)
    
    # Interface de chat principale
    if rag_system is None:
        return
    
    # Initialisation de l'historique
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant", 
            "content": f"👋 Bonjour ! Je suis votre assistant IA local. Je peux répondre à vos questions sur le document **{os.path.basename(PDF_PATH)}** en utilisant **mxbai-embed-large** pour la recherche et **llama3.2** pour la génération. Posez-moi une question !"
        }]
    
    # Affichage de l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input utilisateur
    if prompt := st.chat_input("💭 Posez votre question sur le document..."):
        # Ajout du message utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Affichage du message utilisateur
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Génération de la réponse
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🔄 **Recherche dans le document...**")
            
            try:
                # Ajouter un petit délai pour l'UX
                time.sleep(0.5)
                
                # Génération de la réponse
                full_response = rag_system.ask_question(prompt)
                
                # Affichage de la réponse
                message_placeholder.markdown(full_response)
                
                # Ajout à l'historique
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response
                })
                
            except Exception as e:
                error_msg = f"❌ **Erreur:** {str(e)}"
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })

if __name__ == "__main__":
    main()