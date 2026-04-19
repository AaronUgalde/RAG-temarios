"""
===============================================================================
MÓDULO: rag_engine.py
PROPÓSITO: Motor principal del sistema RAG (Retrieval-Augmented Generation)

Este módulo es el corazón del proyecto. Se encarga de:
1. Convertir los fragmentos de texto en vectores numéricos (embeddings)
2. Almacenar esos vectores en una base de datos vectorial (FAISS)
3. Ante una consulta del usuario, buscar los fragmentos más relevantes
4. Enviar esos fragmentos + la consulta a un LLM vía OpenRouter

CONCEPTO RAG - FLUJO COMPLETO:
┌─────────────────────────────────────────────────────────────────────────┐
│  FASE 1 - INDEXACIÓN (se hace una sola vez al cargar los documentos):   │
│  PDF → Texto → Fragmentos → Embeddings → FAISS Index                    │
│                                                                          │
│  FASE 2 - CONSULTA (ocurre con cada pregunta del usuario):              │
│  Consulta → Embedding → Búsqueda en FAISS → Top-K Fragmentos           │
│           → Prompt enriquecido → LLM (OpenRouter) → Respuesta           │
└─────────────────────────────────────────────────────────────────────────┘

¿Qué es un Embedding?
  Es una representación numérica de un texto (un vector de números).
  Textos con significados similares producen vectores muy parecidos.
  Esto permite hacer búsqueda SEMÁNTICA, no sólo por palabras clave.

¿Qué es FAISS?
  Facebook AI Similarity Search. Librería ultra-eficiente para buscar
  los vectores más similares dentro de una gran colección.

¿Qué es OpenRouter?
  Plataforma que unifica el acceso a cientos de modelos de lenguaje
  (Claude, GPT, Gemini, LLaMA, etc.) bajo una sola API compatible
  con el estándar de OpenAI. Permite cambiar de modelo sin modificar
  el código, solo cambiando el nombre del modelo en la configuración.
===============================================================================
"""

import os
import pickle
import numpy as np
import faiss
import httpx

from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer

# OpenRouter es compatible con el SDK oficial de OpenAI.
# Se usa el mismo cliente, apuntando a la URL base de OpenRouter.
from openai import OpenAI

from document_loader import DocumentLoader


# =============================================================================
# CLASE PRINCIPAL: RAGEngine
# =============================================================================

class RAGEngine:
    """
    Motor RAG que combina búsqueda semántica con generación de lenguaje
    a través de OpenRouter.

    Atributos:
        embedding_model:  Modelo local para convertir texto en vectores.
        index:            Índice FAISS con todos los vectores almacenados.
        chunks:           Lista de fragmentos de texto con su metadata.
        client:           Cliente OpenAI apuntando a la API de OpenRouter.
        model:            Identificador del modelo LLM a utilizar.
        top_k:            Número de fragmentos a recuperar por consulta.
    """

    # Modelo de embeddings local: rápido, gratuito, ~90MB al descargar
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

    # Rutas para persistir el índice entre sesiones
    INDEX_FILE  = "vectorstore/faiss_index.bin"
    CHUNKS_FILE = "vectorstore/chunks_metadata.pkl"

    # URL base de OpenRouter (compatible con el estándar OpenAI)
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, model: str, top_k: int = 5):
        """
        Inicializa el motor RAG configurando el cliente de OpenRouter
        y el modelo de embeddings local.

        Args:
            api_key: Clave de la API de OpenRouter.
            model:   Nombre del modelo en formato OpenRouter
                     (ej: "anthropic/claude-3-haiku", "openai/gpt-4o-mini").
            top_k:   Cuántos fragmentos recuperar por cada consulta.
        """
        print("[RAGEngine] Cargando modelo de embeddings local...")
        # El modelo se descarga la primera vez y se cachea automáticamente
        self.embedding_model = SentenceTransformer(self.EMBEDDING_MODEL_NAME)

        # Configurar el cliente de OpenAI apuntando a OpenRouter.
        # Este patrón es el recomendado por OpenRouter: misma librería,
        # distinta base_url y api_key.
        self.client = OpenAI(
            base_url=self.OPENROUTER_BASE_URL,
            api_key=api_key,
            http_client=httpx.Client(),
        )
        self.model = model
        self.top_k = top_k

        self.index: Optional[faiss.IndexFlatL2] = None
        self.chunks: List[Dict[str, Any]] = []

        print(f"[RAGEngine] Motor inicializado. Modelo LLM: '{self.model}'")

    # =========================================================================
    # FASE 1: INDEXACIÓN - Construir la base vectorial
    # =========================================================================

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Construye el índice FAISS a partir de los fragmentos de texto.

        PROCESO INTERNO:
        1. Extraer el texto de cada fragmento
        2. Pasarlo por el modelo de embeddings → vector de 384 dimensiones
        3. Agregar todos los vectores a un índice FAISS
        4. Persistir el índice en disco para reutilizarlo

        Args:
            chunks: Lista de dicts con "text" y metadata asociada.
        """
        if not chunks:
            raise ValueError("No se puede construir el índice: lista de fragmentos vacía.")

        self.chunks = chunks
        print(f"[RAGEngine] Generando embeddings para {len(chunks)} fragmentos...")

        # Extraer sólo los textos (el modelo sólo acepta strings)
        texts = [chunk["text"] for chunk in chunks]

        # Generar embeddings en lote con barra de progreso
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Normalizar vectores: convierte similitud coseno en distancia L2.
        # Mejora la precisión de búsqueda para texto en lenguaje natural.
        faiss.normalize_L2(embeddings)

        # Dimensión del vector (384 para all-MiniLM-L6-v2)
        embedding_dim = embeddings.shape[1]

        # IndexFlatL2: índice de fuerza bruta con distancia euclidiana.
        # Es el más preciso (compara la consulta contra TODOS los vectores).
        # Para colecciones muy grandes (>100k chunks), considerar IndexIVFFlat.
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.index.add(embeddings)

        print(f"[RAGEngine] Índice construido: {self.index.ntotal} vectores.")
        self._save_index()

    def _save_index(self) -> None:
        """Persiste el índice FAISS y la metadata en disco."""
        os.makedirs("vectorstore", exist_ok=True)
        faiss.write_index(self.index, self.INDEX_FILE)
        with open(self.CHUNKS_FILE, "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"[RAGEngine] Índice guardado en '{self.INDEX_FILE}'")

    def load_index(self) -> bool:
        """
        Carga un índice FAISS previamente guardado en disco.

        Returns:
            True si el índice existe y se cargó; False si no existe.
        """
        if not os.path.exists(self.INDEX_FILE) or not os.path.exists(self.CHUNKS_FILE):
            return False

        self.index = faiss.read_index(self.INDEX_FILE)
        with open(self.CHUNKS_FILE, "rb") as f:
            self.chunks = pickle.load(f)
        print(f"[RAGEngine] Índice cargado: {self.index.ntotal} vectores.")
        return True

    def is_index_built(self) -> bool:
        """Verifica si el índice está listo para consultas."""
        return self.index is not None and len(self.chunks) > 0

    # =========================================================================
    # FASE 2: RECUPERACIÓN - Buscar fragmentos semánticamente relevantes
    # =========================================================================

    def retrieve(self, query: str, carrera_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Busca los fragmentos más relevantes para la consulta dada.

        PROCESO:
        1. Convertir la consulta en vector con el mismo modelo de embeddings
        2. Buscar los top_k vectores más cercanos (mayor similitud semántica)
        3. Aplicar filtro de carrera si se especificó
        4. Retornar los fragmentos con su metadata y score de relevancia

        Args:
            query:          Pregunta o consulta del usuario.
            carrera_filter: Si se especifica, filtra por carrera.

        Returns:
            Lista de fragmentos relevantes con metadata.
        """
        if not self.is_index_built():
            raise RuntimeError("El índice no está construido. Cargue documentos primero.")

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )
        faiss.normalize_L2(query_embedding)

        # Buscar más resultados si hay filtro (algunos serán descartados)
        search_k = self.top_k * 3 if carrera_filter else self.top_k
        distances, indices = self.index.search(query_embedding, search_k)

        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx].copy()
            # Convertir distancia a score 0-1 (mayor = más relevante)
            chunk["relevance_score"] = float(1 / (1 + distance))

            if carrera_filter:
                if carrera_filter.lower() in chunk["carrera"].lower():
                    results.append(chunk)
            else:
                results.append(chunk)

            if len(results) >= self.top_k:
                break

        return results

    # =========================================================================
    # FASE 3: GENERACIÓN - Responder usando el LLM con contexto recuperado
    # =========================================================================

    def generate_response(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict]] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Genera una respuesta usando el LLM configurado en OpenRouter,
        enriquecida con los fragmentos recuperados de los temarios.

        CONCEPTO CLAVE - PROMPT ENGINEERING PARA RAG:
        El prompt incluye tres partes fundamentales:
          1. System prompt: define el rol y las restricciones del asistente
          2. Contexto: fragmentos relevantes recuperados de los documentos
          3. Consulta: la pregunta concreta del usuario

        Esto le permite al LLM generar respuestas precisas y verificables,
        en lugar de depender únicamente de su conocimiento preentrenado.

        Args:
            query:                Pregunta del usuario.
            retrieved_chunks:     Fragmentos relevantes del índice FAISS.
            conversation_history: Historial de mensajes previos (multi-turno).

        Returns:
            Tupla (texto_de_respuesta, lista_de_fragmentos_usados)
        """
        # --- Construir el bloque de contexto ---
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(
                f"[Fuente {i}] Carrera: {chunk['carrera']} | "
                f"Archivo: {chunk['source']} | Página: {chunk['page']}\n"
                f"{chunk['text']}"
            )
        context_block = "\n\n---\n\n".join(context_parts)

        # --- System prompt: rol y restricciones del asistente ---
        system_prompt = (
            "Eres un asistente académico especializado en planes de estudio y "
            "temarios universitarios. Tu función es ayudar a estudiantes, docentes "
            "y coordinadores a consultar información sobre los contenidos de diversas "
            "carreras universitarias.\n\n"
            "INSTRUCCIONES:\n"
            "1. Responde ÚNICAMENTE con la información del contexto proporcionado.\n"
            "2. Si la respuesta no está en el contexto, indícalo explícitamente.\n"
            "3. Cita siempre la carrera y fuente de donde proviene la información.\n"
            "4. Usa lenguaje académico claro y estructurado.\n"
            "5. Al comparar múltiples carreras, organiza la información en tablas o listas."
        )

        # --- Mensaje del usuario con el contexto incrustado ---
        user_message = (
            f"Fragmentos relevantes de los temarios universitarios:\n\n"
            f"{context_block}\n\n"
            f"---\n\n"
            f"Consulta: {query}"
        )

        # --- Construir el historial de mensajes ---
        # La API de OpenRouter (formato OpenAI) espera una lista de mensajes:
        # [{"role": "system"|"user"|"assistant", "content": "..."}]
        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_message})

        # --- Llamar a la API de OpenRouter ---
        # Se usa el SDK de openai con base_url apuntando a OpenRouter.
        # El parámetro `model` acepta cualquier modelo de openrouter.ai/models
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1500,
            messages=messages,
        )

        answer_text = response.choices[0].message.content
        return answer_text, retrieved_chunks

    # =========================================================================
    # MÉTODO PRINCIPAL: query (Retrieve + Generate en un solo paso)
    # =========================================================================

    def query(
        self,
        user_question: str,
        carrera_filter: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Método principal del motor RAG: recibe una pregunta del usuario
        y retorna una respuesta fundamentada en los temarios.

        Args:
            user_question:        Pregunta en lenguaje natural.
            carrera_filter:       Carrera para filtrar resultados (opcional).
            conversation_history: Historial del chat actual.

        Returns:
            Dict con "answer" (str) y "sources" (list de fragmentos).
        """
        retrieved = self.retrieve(user_question, carrera_filter)

        if not retrieved:
            return {
                "answer": (
                    "No encontré información relevante en los temarios disponibles. "
                    "Reformule su consulta o verifique que los documentos estén cargados."
                ),
                "sources": []
            }

        answer, sources = self.generate_response(
            query=user_question,
            retrieved_chunks=retrieved,
            conversation_history=conversation_history
        )

        return {"answer": answer, "sources": sources}

    # =========================================================================
    # UTILIDADES
    # =========================================================================

    def get_index_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del índice vectorial actual."""
        if not self.is_index_built():
            return {"status": "No hay índice construido."}

        carreras = sorted(set(c["carrera"] for c in self.chunks))
        fuentes  = sorted(set(c["source"]  for c in self.chunks))

        return {
            "total_fragmentos":    len(self.chunks),
            "total_vectores_faiss": self.index.ntotal,
            "carreras":            carreras,
            "archivos":            fuentes,
        }
