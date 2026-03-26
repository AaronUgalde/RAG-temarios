"""
===============================================================================
MÓDULO: app.py
PROPÓSITO: Interfaz de usuario web para el sistema RAG de Temarios

Este es el punto de entrada de la aplicación. Utiliza Streamlit para crear
una interfaz gráfica que permite:
  - Cargar y visualizar los temarios disponibles
  - Construir o recargar el índice vectorial
  - Hacer consultas en lenguaje natural sobre los temarios
  - Ver las fuentes exactas de donde proviene la información

CÓMO EJECUTAR:
  streamlit run app.py

ESTRUCTURA DE LA INTERFAZ:
  ┌──────────────────────────────────────────────────────┐
  │  Barra lateral (sidebar):                            │
  │    - Estado del índice                               │
  │    - Carreras disponibles                            │
  │    - Filtro por carrera                              │
  │    - Botón para reindexar                            │
  ├──────────────────────────────────────────────────────┤
  │  Panel principal:                                    │
  │    - Historial de chat                               │
  │    - Campo de entrada de consultas                   │
  │    - Respuestas con fuentes citadas                  │
  └──────────────────────────────────────────────────────┘
===============================================================================
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Importar nuestros módulos del proyecto
from document_loader import DocumentLoader
from rag_engine import RAGEngine


# =============================================================================
# CONFIGURACIÓN INICIAL DE LA APLICACIÓN
# =============================================================================

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="RAG Temarios Universitarios",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Rutas y parámetros del sistema (leídos desde .env) ---
TEMARIOS_PATH      = "temarios"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3-haiku")
RAG_TOP_K          = int(os.getenv("RAG_TOP_K", "5"))
CHUNK_SIZE         = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP      = int(os.getenv("CHUNK_OVERLAP", "200"))


# =============================================================================
# INICIALIZACIÓN DEL ESTADO DE SESIÓN (st.session_state)
# =============================================================================
# Streamlit re-ejecuta el script completo en cada interacción del usuario.
# Para mantener datos entre interacciones, se usa st.session_state, que
# funciona como un diccionario persistente durante la sesión del usuario.

def initialize_session_state():
    """
    Inicializa las variables de estado de la sesión si aún no existen.
    Se llama una sola vez al inicio de cada sesión de usuario.
    """
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None

    if "index_built" not in st.session_state:
        st.session_state.index_built = False

    # Historial del chat: lista de dicts {"role": "user"|"assistant", "content": str}
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Fuentes del último mensaje del asistente
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []


# =============================================================================
# FUNCIONES DE INICIALIZACIÓN DEL MOTOR RAG
# =============================================================================

@st.cache_resource  # Streamlit cachea este recurso: no se recrea en cada interacción
def get_rag_engine(api_key: str, model: str, top_k: int) -> RAGEngine:  # noqa: E501
    """
    Crea e inicializa el motor RAG (se ejecuta una sola vez por sesión).
    El decorador @st.cache_resource garantiza que el modelo de embeddings
    no se recargue innecesariamente.

    Args:
        api_key: Clave de la API de Anthropic.
        model:   Nombre del modelo Claude.
        top_k:   Número de fragmentos a recuperar por consulta.

    Returns:
        Instancia de RAGEngine lista para usar.
    """
    return RAGEngine(api_key=api_key, model=model, top_k=top_k)


def build_or_load_index(engine: RAGEngine) -> bool:
    """
    Intenta cargar un índice guardado; si no existe, construye uno nuevo.

    Este flujo optimiza el tiempo de inicio: si ya se indexaron los documentos
    en una sesión anterior, no es necesario volver a procesar los PDFs.

    Args:
        engine: Instancia del motor RAG.

    Returns:
        True si el índice quedó listo, False si no hay documentos.
    """
    # Intentar cargar un índice previamente guardado
    if engine.load_index():
        return True

    # Si no existe índice guardado, procesar los PDFs y construir uno nuevo
    loader = DocumentLoader(
        temarios_path=TEMARIOS_PATH,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = loader.load_all_documents()

    if not chunks:
        return False

    engine.build_index(chunks)
    return True


# =============================================================================
# COMPONENTES DE LA INTERFAZ
# =============================================================================

def render_sidebar(engine: RAGEngine) -> str:
    """
    Renderiza la barra lateral de la aplicación.

    Args:
        engine: Motor RAG inicializado.

    Returns:
        Nombre de la carrera seleccionada para filtrar (o cadena vacía).
    """
    st.sidebar.title("🎓 Configuración")
    st.sidebar.markdown("---")

    # --- Estado del índice ---
    st.sidebar.subheader("📊 Estado del Sistema")
    if st.session_state.index_built:
        stats = engine.get_index_stats()
        st.sidebar.success(f"✅ Índice activo")
        st.sidebar.metric("Fragmentos indexados", stats.get("total_fragmentos", 0))
        st.sidebar.metric("Carreras disponibles", len(stats.get("carreras", [])))
    else:
        st.sidebar.warning("⚠️ Índice no construido")

    st.sidebar.markdown("---")

    # --- Filtro por carrera ---
    carrera_filter = ""
    if st.session_state.index_built:
        stats = engine.get_index_stats()
        carreras = ["Todas las carreras"] + stats.get("carreras", [])

        st.sidebar.subheader("🔍 Filtrar por Carrera")
        seleccion = st.sidebar.selectbox(
            "Seleccione una carrera:",
            options=carreras,
            help="Filtra las respuestas para que sólo usen documentos de la carrera seleccionada."
        )
        if seleccion != "Todas las carreras":
            carrera_filter = seleccion

    st.sidebar.markdown("---")

    # --- Botón para reindexar ---
    st.sidebar.subheader("🔄 Gestión del Índice")
    if st.sidebar.button("Reindexar documentos", help="Vuelve a leer todos los PDFs y reconstruye el índice."):
        with st.spinner("Procesando documentos y construyendo el índice..."):
            loader = DocumentLoader(TEMARIOS_PATH, CHUNK_SIZE, CHUNK_OVERLAP)
            chunks = loader.load_all_documents()
            if chunks:
                engine.build_index(chunks)
                st.session_state.index_built = True
                st.sidebar.success("Índice reconstruido exitosamente.")
                st.rerun()
            else:
                st.sidebar.error(f"No se encontraron PDFs en la carpeta '{TEMARIOS_PATH}/'")

    # --- Información del proyecto ---
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "**RAG Temarios** v1.0\n\n"
        "Coloque sus archivos PDF en la carpeta `temarios/` "
        "y presione *Reindexar documentos* para comenzar."
    )

    return carrera_filter


def render_chat(engine: RAGEngine, carrera_filter: str):
    """
    Renderiza el panel principal del chat con historial y campo de entrada.

    Args:
        engine:          Motor RAG para procesar las consultas.
        carrera_filter:  Carrera seleccionada en el sidebar (puede ser vacía).
    """
    # --- Encabezado principal ---
    st.title("🎓 Consultor de Temarios Universitarios")
    st.caption(
        "Haga preguntas sobre los planes de estudio, materias, contenidos y "
        "requisitos de las carreras disponibles."
    )

    if not st.session_state.index_built:
        # Mostrar instrucciones cuando no hay índice
        st.info(
            "**Para comenzar:**\n\n"
            "1. Coloque archivos PDF de temarios en la carpeta `temarios/`\n"
            "2. Presione **Reindexar documentos** en la barra lateral\n"
            "3. ¡Comience a hacer consultas!"
        )
        return

    # --- Mostrar historial del chat ---
    # Se itera sobre los mensajes guardados en session_state
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Mostrar las fuentes sólo para los mensajes del asistente
            if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                render_sources(msg["sources"])

    # --- Campo de entrada de consultas ---
    # st.chat_input es un campo de texto que se limpia automáticamente al enviar
    if user_input := st.chat_input("Escriba su consulta sobre los temarios..."):

        # Agregar el mensaje del usuario al historial
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # Mostrar el mensaje del usuario en la interfaz
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generar y mostrar la respuesta del asistente
        with st.chat_message("assistant"):
            with st.spinner("Consultando los temarios..."):

                # Preparar el historial para la API de Claude
                # (sólo los intercambios anteriores, sin el último mensaje del usuario)
                history_for_api = []
                for msg in st.session_state.messages[:-1]:  # Excluir el último
                    history_for_api.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

                # Consultar el motor RAG
                result = engine.query(
                    user_question=user_input,
                    carrera_filter=carrera_filter if carrera_filter else None,
                    conversation_history=history_for_api if history_for_api else None
                )

            # Mostrar la respuesta
            st.markdown(result["answer"])

            # Mostrar las fuentes utilizadas
            if result["sources"]:
                render_sources(result["sources"])

        # Guardar la respuesta en el historial de sesión
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })

    # --- Botón para limpiar el historial ---
    if st.session_state.messages:
        col1, col2 = st.columns([8, 2])
        with col2:
            if st.button("🗑️ Limpiar chat"):
                st.session_state.messages = []
                st.rerun()


def render_sources(sources: list):
    """
    Renderiza un expander con las fuentes utilizadas para la respuesta.

    Mostrar las fuentes es fundamental en un sistema RAG para que el usuario
    pueda verificar la información y saber de qué documentos proviene.

    Args:
        sources: Lista de fragmentos utilizados como contexto.
    """
    with st.expander(f"📚 Fuentes consultadas ({len(sources)} fragmentos)"):
        for i, source in enumerate(sources, 1):
            st.markdown(
                f"**[{i}] {source['carrera']}** — "
                f"`{source['source']}` — "
                f"Página {source['page']} — "
                f"Relevancia: `{source.get('relevance_score', 0):.2f}`"
            )
            st.caption(source['text'][:300] + "..." if len(source['text']) > 300 else source['text'])
            if i < len(sources):
                st.divider()


# =============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# =============================================================================

def main():
    """
    Función principal que orquesta toda la aplicación Streamlit.
    Esta función se ejecuta completa en cada interacción del usuario.
    """
    # Paso 1: Inicializar variables de estado de la sesión
    initialize_session_state()

    # Paso 2: Validar que la API key esté configurada
    if not OPENROUTER_API_KEY:
        st.error(
            "**Error de configuración:** No se encontró la clave de API de OpenRouter.\n\n"
            "1. Copie el archivo `.env.example` y renómbrelo a `.env`\n"
            "2. Agregue su clave: `OPENROUTER_API_KEY=sk-or-v1-...`\n"
            "3. Obtenga su clave en: https://openrouter.ai/keys\n"
            "4. Reinicie la aplicación"
        )
        st.stop()

    # Paso 3: Obtener (o crear) el motor RAG
    engine = get_rag_engine(OPENROUTER_API_KEY, OPENROUTER_MODEL, RAG_TOP_K)
    st.session_state.rag_engine = engine

    # Paso 4: Cargar o construir el índice (sólo si aún no está en session_state)
    if not st.session_state.index_built:
        with st.spinner("Cargando el índice vectorial..."):
            if build_or_load_index(engine):
                st.session_state.index_built = True

    # Paso 5: Renderizar la barra lateral y obtener el filtro de carrera
    carrera_filter = render_sidebar(engine)

    # Paso 6: Renderizar el panel principal del chat
    render_chat(engine, carrera_filter)


# Punto de entrada estándar de Python
if __name__ == "__main__":
    main()
