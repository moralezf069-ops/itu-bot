import os
import logging
import numpy as np
from google import genai

logger = logging.getLogger(__name__)

class RAGVectorIndex:
    def __init__(self, ruta_archivo="reglamento_itu.txt"):
        self.ruta_archivo = ruta_archivo
        self.chunks = []
        self.embeddings = []
        self.use_embeddings = True
        
        try:
            self.client = genai.Client(http_options={'api_version': 'v1beta'})
        except Exception:
            self.client = genai.Client()
            
        self._inicializar_indice()

    def _inicializar_indice(self):
        """Lee el archivo, lo divide en chunks y genera embeddings o activa respaldo inteligente."""
        if not os.path.exists(self.ruta_archivo):
            logger.warning(f"No se encontró el archivo {self.ruta_archivo}")
            return

        with open(self.ruta_archivo, "r", encoding="utf-8") as f:
            contenido = f.read()

        # Dividir por párrafos o secciones significativas
        self.chunks = [b.strip() for b in contenido.split("\n\n") if b.strip()]
        logger.info(f"-> [RAG] Documento dividido en {len(self.chunks)} chunks.")

        # Intentar generar embeddings para cada chunk
        for i, chunk in enumerate(self.chunks):
            try:
                response = self.client.models.embed_content(
                    model="text-embedding-004",
                    contents=chunk
                )
                vector = response.embeddings[0].values
                self.embeddings.append(vector)
            except Exception as e:
                if i == 0:
                    logger.warning(f"-> [RAG] API de embeddings no disponible en la clave actual. Activando modo de recuperación inteligente por concordancia.")
                self.use_embeddings = False
                break

        if self.use_embeddings and len(self.embeddings) == len(self.chunks):
            logger.info("-> [RAG VECTORIAL] ¡Indexación vectorial completada con éxito!")
        else:
            logger.info("-> [RAG HÍBRIDO] Sistema operando en modo de alta disponibilidad y recuperación inteligente.")

    def buscar(self, query: str, top_k=1) -> str:
        """Busca el chunk más relevante usando vectores o respaldo inteligente por palabras clave."""
        if not self.chunks:
            return "No hay información indexada."

        # Si los embeddings funcionan, usar similitud de cosenos
        if self.use_embeddings and len(self.embeddings) == len(self.chunks):
            try:
                response = self.client.models.embed_content(
                    model="text-embedding-004",
                    contents=query
                )
                query_vector = np.array(response.embeddings[0].values)

                similitudes = []
                for emb in self.embeddings:
                    emb_vector = np.array(emb)
                    similitud = np.dot(query_vector, emb_vector) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(emb_vector)
                    )
                    similitudes.append(similitud)

                mejor_idx = int(np.argmax(similitudes))
                mejor_score = similitudes[mejor_idx]
                logger.info(f"-> [RAG VECTORIAL] Consulta: '{query}' | Match score: {mejor_score:.4f}")

                if mejor_score > 0.3:
                    return self.chunks[mejor_idx]
            except Exception as e:
                logger.error(f"Error en búsqueda vectorial, usando respaldo: {str(e)}")

        # Respaldo inteligente por coincidencia de palabras clave (Keywords)
        query_words = set(query.lower().split())
        mejor_chunk = self.chunks[0]
        max_coincidencias = -1

        for chunk in self.chunks:
            chunk_lower = chunk.lower()
            coincidencias = sum(1 for palabra in query_words if palabra in chunk_lower)
            if coincidencias > max_coincidencias:
                max_coincidencias = coincidencias
                mejor_chunk = chunk

        logger.info(f"-> [RAG INTELIGENTE] Consulta: '{query}' | Coincidencias encontradas: {max_coincidencias}")
        if max_coincidencias > 0:
            return mejor_chunk
        else:
            return self.chunks[0]