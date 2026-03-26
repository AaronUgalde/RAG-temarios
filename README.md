# RAG Temarios Universitarios

Sistema de Retrieval-Augmented Generation (RAG) para consultar planes de estudio
y temarios universitarios mediante lenguaje natural.

---

## Descripcion del sistema

RAG es una arquitectura de inteligencia artificial que combina dos capacidades:

**Recuperacion (Retrieval):** Ante una consulta del usuario, el sistema busca los
fragmentos de texto mas relevantes dentro de los documentos indexados, utilizando
busqueda semantica vectorial mediante FAISS y Sentence Transformers.

**Generacion (Generation):** Los fragmentos recuperados se inyectan como contexto
en el prompt de un modelo de lenguaje (LLM), que genera una respuesta coherente,
precisa y fundamentada en los documentos originales.

Flujo del sistema:

```
INDEXACION (una sola vez):
PDF --> Texto --> Fragmentos (chunks) --> Embeddings --> Indice FAISS

CONSULTA (por cada pregunta):
Pregunta --> Embedding --> Busqueda FAISS --> Top-K Fragmentos
         --> Prompt enriquecido --> LLM via OpenRouter --> Respuesta
```

---

## Estructura del proyecto

```
RAG temarios/
|-- app.py               Interfaz web (Streamlit). Punto de entrada.
|-- rag_engine.py        Motor RAG: embeddings, FAISS y generacion via OpenRouter.
|-- document_loader.py   Carga, extraccion y fragmentacion de archivos PDF.
|-- requirements.txt     Dependencias del proyecto.
|-- .env.example         Plantilla de configuracion de variables de entorno.
|-- .gitignore
|-- temarios/            Directorio donde se depositan los archivos PDF.
|   `-- (archivos .pdf)
`-- vectorstore/         Indice FAISS generado automaticamente (no subir a Git).
    |-- faiss_index.bin
    `-- chunks_metadata.pkl
```

---

## Instalacion

### Requisitos previos

- Python 3.12 o superior
- Cuenta en OpenRouter: https://openrouter.ai

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

La primera ejecucion descarga el modelo de embeddings `all-MiniLM-L6-v2`
(aproximadamente 90 MB). Este modelo se almacena en cache de forma local.

### 2. Configurar variables de entorno

```bash
copy .env.example .env
```

Editar el archivo `.env` y completar los valores requeridos:

```
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=anthropic/claude-3-haiku
```

La clave de API se obtiene en: https://openrouter.ai/keys

### 3. Agregar documentos

Copiar los archivos PDF de temarios en la carpeta `temarios/`.

Convencion de nombres recomendada:

```
temarios/
|-- temario_ingenieria_sistemas.pdf
|-- temario_administracion_empresas.pdf
`-- temario_medicina.pdf
```

El nombre de la carrera se infiere automaticamente a partir del nombre del archivo,
eliminando prefijos comunes como "temario", "plan" o "programa".

### 4. Ejecutar la aplicacion

```bash
streamlit run app.py
```

La aplicacion estara disponible en: http://localhost:8501

---

## Uso

Al iniciar, el sistema intenta cargar un indice previamente construido desde el
directorio `vectorstore/`. Si no existe, es necesario construirlo manualmente.

**Para construir o reconstruir el indice:**
Presionar el boton "Reindexar documentos" en la barra lateral. Este proceso
lee todos los PDFs de la carpeta `temarios/`, extrae el texto, lo divide en
fragmentos y genera los embeddings vectoriales.

**Filtro por carrera:**
Es posible restringir las consultas a una carrera especifica mediante el
selector de la barra lateral. Si se selecciona "Todas las carreras", el
sistema busca en todos los documentos indexados.

**Ejemplos de consultas:**

- "Que materias de matematicas incluye la carrera de Ingenieria en Sistemas?"
- "Cuantos semestres dura la carrera de Administracion de Empresas?"
- "Compara las materias de programacion entre Sistemas y Ciencia de Datos."
- "Que asignaturas optativas ofrece la carrera de Actuaria?"

---

## Modelos disponibles en OpenRouter

OpenRouter permite acceder a multiples modelos bajo una misma API.
El modelo se configura en el archivo `.env` mediante la variable `OPENROUTER_MODEL`.

Ejemplos de modelos disponibles:

| Identificador                          | Descripcion                        |
|----------------------------------------|------------------------------------|
| `anthropic/claude-3-haiku`             | Rapido y economico                 |
| `anthropic/claude-3.5-sonnet`          | Alta calidad de respuesta          |
| `openai/gpt-4o-mini`                   | Equilibrado en costo y calidad     |
| `google/gemini-flash-1.5`             | Muy rapido                         |
| `meta-llama/llama-3.1-8b-instruct`    | Codigo abierto, con limite gratuito|

Catalogo completo en: https://openrouter.ai/models

---

## Tecnologias utilizadas

| Componente              | Tecnologia                                  |
|-------------------------|---------------------------------------------|
| Interfaz web            | Streamlit                                   |
| Modelo de lenguaje      | Cualquier LLM disponible via OpenRouter     |
| Cliente de API          | SDK de OpenAI (compatible con OpenRouter)   |
| Embeddings              | all-MiniLM-L6-v2 (Sentence Transformers)    |
| Base de datos vectorial | FAISS (Facebook AI Similarity Search)       |
| Lectura de PDFs         | PyPDF                                       |

---

## Notas tecnicas

**Embeddings:** Los embeddings son representaciones numericas del texto que
capturan su significado semantico. Textos con significados similares producen
vectores cercanos en el espacio vectorial, lo que permite busqueda por
similitud de significado en lugar de coincidencia de palabras exactas.

**Chunk overlap:** La superposicion entre fragmentos consecutivos garantiza
que el contexto no se pierda en los bordes de cada fragmento. Con los valores
por defecto (`CHUNK_SIZE=1000`, `CHUNK_OVERLAP=200`), cada fragmento comparte
200 caracteres con el fragmento anterior y el siguiente.

**Persistencia del indice:** El indice FAISS y la metadata de fragmentos se
guardan en disco al construirse. En ejecuciones posteriores se cargan
directamente sin reprocesar los PDFs, lo que reduce el tiempo de inicio.
Es necesario reconstruir el indice cada vez que se agreguen o modifiquen
documentos en la carpeta `temarios/`.

**Seguridad:** El archivo `.env` contiene la clave de API y no debe subirse
a repositorios de codigo. El archivo `.gitignore` incluido ya lo excluye.
