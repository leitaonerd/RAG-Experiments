# app.py
import ollama
import PyPDF2
import os
import numpy as np
import json
import hashlib
from datetime import datetime
from sentence_transformers import SentenceTransformer
from flask import Flask, render_template, request, jsonify, session
from flask_caching import Cache

app = Flask(__name__)
app.secret_key = 'sua_key_privada'  # se for usar outa coisa alem do ollama
app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)

class PDFChatSystem:
    def __init__(self, pdf_folder="./pdfs/"):
        self.pdf_folder = pdf_folder
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.document_metadata = []
        self.embeddings = None
        self.embedding_cache = {}
        self.cache_file = "embedding_cache.json"
        
        # Carregar cache existente se disponível
        self.load_embedding_cache()
    
    def load_embedding_cache(self):
        """Carrega o cache de embeddings do arquivo"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.embedding_cache = json.load(f)
                print(f"Cache carregado com {len(self.embedding_cache)} entradas")
        except Exception as e:
            print(f"Erro ao carregar cache: {e}")
            self.embedding_cache = {}
    
    def save_embedding_cache(self):
        """Salva o cache de embeddings no arquivo"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.embedding_cache, f, ensure_ascii=False, indent=2)
            print(f"Cache salvo com {len(self.embedding_cache)} entradas")
        except Exception as e:
            print(f"Erro ao salvar cache: {e}")
    
    def get_text_hash(self, text):
        """Gera um hash único para o texto"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_cached_embedding(self, text):
        """Recupera embedding do cache se disponível"""
        text_hash = self.get_text_hash(text)
        if text_hash in self.embedding_cache:
            # Converte de volta para numpy array
            return np.array(self.embedding_cache[text_hash]['embedding'])
        return None
    
    def cache_embedding(self, text, embedding):
        """Armazena embedding no cache"""
        text_hash = self.get_text_hash(text)
        # Converte numpy array para lista para serialização JSON
        self.embedding_cache[text_hash] = {
            'embedding': embedding.tolist(),
            'timestamp': datetime.now().isoformat()
        }
    
    def extract_text_from_pdf(self, pdf_path):
        """Extrai texto de um arquivo PDF"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n--- Página {i+1} ---\n{page_text}"
        except Exception as e:
            print(f"Erro ao extrair texto de {pdf_path}: {e}")
        return text
    
    def load_documents(self):
        """Carrega todos os PDFs da pasta especificada"""
        self.documents = []
        self.document_metadata = []
        
        if not os.path.exists(self.pdf_folder):
            os.makedirs(self.pdf_folder)
            print(f"Pasta {self.pdf_folder} criada. Adicione seus PDFs lá.")
            return
        
        pdf_files = [f for f in os.listdir(self.pdf_folder) if f.endswith('.pdf')]
        
        if not pdf_files:
            print(f"Nenhum PDF encontrado em {self.pdf_folder}")
            return
        
        print(f"Encontrados {len(pdf_files)} arquivos PDF")
        
        for filename in pdf_files:
            print(f"Processando: {filename}")
            pdf_path = os.path.join(self.pdf_folder, filename)
            text = self.extract_text_from_pdf(pdf_path)
            
            if not text.strip():
                print(f"Aviso: {filename} parece estar vazio ou não pôde ser lido")
                continue
            
            # Divide o texto em chunks menores
            chunks = self.split_text_into_chunks(text, chunk_size=1000, overlap=200)
            
            for i, chunk in enumerate(chunks):
                self.documents.append(chunk)
                self.document_metadata.append({
                    'file': filename,
                    'chunk': i,
                    'pages': 'Várias'  # Simplificado para este exemplo
                })
        
        print(f"Total de {len(self.documents)} chunks de texto extraídos")
        
        # Gera embeddings para todos os documentos
        self.generate_embeddings()
    
    def split_text_into_chunks(self, text, chunk_size=1000, overlap=200):
        """Divide o texto em chunks com sobreposição"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            # Garante que não quebre no meio de uma palavra
            if end < len(text):
                while end > start and text[end] not in ' \n\t.,!?;:':
                    end -= 1
                if end == start:  # Fallback se não encontrar espaço
                    end = start + chunk_size
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap  # Sobrepõe os chunks
            
        return chunks
    
    def generate_embeddings(self):
        """Gera embeddings para todos os documentos, usando cache quando disponível"""
        if not self.documents:
            print("Nenhum documento para gerar embeddings")
            return
        
        print("Gerando embeddings...")
        embeddings_list = []
        documents_to_process = []
        indices_to_process = []
        
        # Verifica quais documentos já estão em cache
        for i, doc in enumerate(self.documents):
            cached_embedding = self.get_cached_embedding(doc)
            if cached_embedding is not None:
                embeddings_list.append(cached_embedding)
            else:
                documents_to_process.append(doc)
                indices_to_process.append(i)
                # Placeholder que será substituído
                embeddings_list.append(np.zeros(384))  # 384 é a dimensão do all-MiniLM-L6-v2
        
        # Processa apenas os documentos não encontrados em cache
        if documents_to_process:
            print(f"Processando {len(documents_to_process)} novos documentos")
            new_embeddings = self.model.encode(documents_to_process)
            
            # Atualiza a lista de embeddings e o cache
            for idx, embedding in zip(indices_to_process, new_embeddings):
                embeddings_list[idx] = embedding
                self.cache_embedding(self.documents[idx], embedding)
            
            # Salva o cache atualizado
            self.save_embedding_cache()
        
        self.embeddings = np.array(embeddings_list)
        print("Embeddings gerados e cache atualizado")
    
    def find_relevant_chunks(self, query, top_k=5):
        """Encontra os chunks mais relevantes para a consulta"""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
        
        # Gera embedding para a consulta (também usa cache)
        cached_query_embedding = self.get_cached_embedding(query)
        if cached_query_embedding is not None:
            query_embedding = cached_query_embedding
        else:
            query_embedding = self.model.encode([query])[0]
            self.cache_embedding(query, query_embedding)
            self.save_embedding_cache()
        
        # Calcula similaridades
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Obtém os índices dos top_k chunks mais relevantes
        indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Prepara os resultados
        results = []
        for idx in indices:
            results.append({
                'text': self.documents[idx],
                'metadata': self.document_metadata[idx],
                'similarity': float(similarities[idx])
            })
        
        return results
    
    def ask_question(self, question, top_k=5):
        """Responde a uma pergunta com base nos documentos"""
        if not self.documents:
            return "Não há documentos carregados. Por favor, adicione PDFs na pasta."
        
        relevant_chunks = self.find_relevant_chunks(question, top_k=top_k)
        
        if not relevant_chunks:
            return "Não encontrei informações relevantes para responder à sua pergunta."
        
        # Prepara o contexto
        context = ""
        for i, chunk in enumerate(relevant_chunks):
            context += f"\n--- Trecho {i+1} (de {chunk['metadata']['file']}) ---\n"
            context += chunk['text'] + "\n"
        
        # Prepara o prompt para o modelo
        prompt = f"""Com base nos trechos de documentos abaixo, responda à pergunta do usuário.
Se a resposta não estiver contida nos documentos, diga explicitamente que não encontrou uma referência direta nos arquivos e, caso 
conheça sobre o assunto, responda.
Sempre dê preferência aos documentos carregados, e sempre que se possa complementar a resposta, faça-o e indique que o complemento foi 
feito com base no seu conhecimento prévio de forma explícita.

Trechos dos documentos:
{context}

Pergunta: {question}

Resposta:"""
        
        try:
            # Chama o Ollama
            response = ollama.chat(model='llama3:8b', messages=[
                {'role': 'user', 'content': prompt}
            ])
            
            # Adiciona as fontes à resposta
            answer = response['message']['content']
            answer += "\n\n---\n*Fontes consultadas:*\n"
            for chunk in relevant_chunks:
                answer += f"- {chunk['metadata']['file']} (páginas {chunk['metadata']['pages']})\n"
            
            return answer
        except Exception as e:
            return f"Erro ao consultar o modelo: {str(e)}"

# Inicializa o sistema
pdf_chat = PDFChatSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load', methods=['POST'])
def load_documents():
    """Rota para carregar documentos"""
    try:
        pdf_chat.load_documents()
        return jsonify({
            'success': True,
            'message': f'Documentos carregados: {len(pdf_chat.documents)} chunks de texto'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Erro ao carregar documentos: {str(e)}'
        })

@app.route('/ask', methods=['POST'])
def ask_question():
    """Rota para fazer perguntas"""
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'Pergunta vazia'})
    
    try:
        answer = pdf_chat.ask_question(question)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': f'Erro ao processar pergunta: {str(e)}'})

@app.route('/cache_info')
def cache_info():
    """Rota para obter informações sobre o cache"""
    return jsonify({
        'cache_size': len(pdf_chat.embedding_cache),
        'documents_loaded': len(pdf_chat.documents)
    })

if __name__ == '__main__':
    # Carrega documentos ao iniciar (opcional)
    # pdf_chat.load_documents()
    app.run(debug=True, port=5000)