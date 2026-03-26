# 🎓 RAG Temarios Universitarios

Sistema de **Retrieval-Augmented Generation (RAG)** para consultar información
sobre planes de estudio y temarios de carreras universitarias en lenguaje natural.

---

## 📐 ¿Qué es RAG?

RAG (Retrieval-Augmented Generation) es una arquitectura de IA que combina dos capacidades:

1. **Recuperación (Retrieval):** Busca fragmentos de texto relevantes en una base de
   documentos usando búsqueda semántica vectorial (FAISS + Sentence Transformers).

2. **Generación (Generation):** Utiliza un modelo de lenguaje grande (Claude de Anthropic)
   para generar una respuesta coherente y precisa, enriquecida con el contexto recuperado.

```
Usuario ──→ Consulta ──→ [Embeddings] ──→ FAISS
                                           │
                                   Top-K Fragmentos
                                           │
                         [Claude + Contexto] ──→ Respuesta
```

---

## 🗂️ Estructura del Proyecto

```
RAG temarios/
├── app.py               # Interfaz web (Streamlit) — punto de entrada
├── rag_engine.py        # Motor RAG: embeddings, FAISS y generación con Claude
├── document_loader.py   # Carga y fragmentación de PDFs
├── requirements.txt     # Dependencias del proyecto
├── .env.example         # Plantilla de configuración
├── .gitignore           # Archivos ignorados por Git
├── temarios/            # ← COLOQUE SUS PDFs AQUÍ
│   └── (archivos .pdf)
└── vectorstore/         # Índice FAISS generado automáticamente
    ├── faiss_index.bin
    └── chunks_metadata.pkl
```

---

## 🚀 Instalación y Configuración

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar variables de entorno

```bash
# Copiar la plantilla
copy .env.example .env

# Editar .env y agregar su clave de API de Anthropic
# ANTHROPIC_API_KEY=sk-ant-...
```

Obtenga su clave de API en: https://console.anthropic.com/

### 3. Agregar documentos

Copie sus archivos PDF de temarios en la carpeta `temarios/`.

**Convención de nombres recomendada:**
```
temarios/
├── temario_ingenieria_sistemas.pdf
├── temario_administracion_empresas.pdf
├── temario_medicina.pdf
└── temario_derecho.pdf
```
El nombre de la carrera se infiere automáticamente del nombre del archivo.

### 4. Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en: http://localhost:8501

---

## 💡 Uso

1. Al iniciar, la aplicación intentará cargar un índice previamente guardado.
2. Si es la primera vez, presione **"Reindexar documentos"** en la barra lateral.
3. Seleccione una carrera para filtrar (opcional).
4. Escriba su consulta en el campo de texto y presione Enter.

### Ejemplos de consultas:
- *"¿Qué materias de matemáticas tiene la carrera de Ingeniería en Sistemas?"*
- *"¿Cuántos semestres dura la carrera de Administración?"*
- *"Compara las materias de programación entre Sistemas y Mecatrónica"*
- *"¿Qué asignaturas optativas ofrece la carrera de Medicina?"*

---

## ⚙️ Tecnologías Utilizadas

| Componente         | Tecnología                          |
|--------------------|-------------------------------------|
| Interfaz web       | Streamlit                           |
| Modelo de lenguaje | Claude (Anthropic)                  |
| Embeddings         | all-MiniLM-L6-v2 (Sentence Transformers) |
| Base vectorial     | FAISS (Facebook AI)                 |
| Lectura de PDFs    | PyPDF                               |

---

## 📝 Notas Educativas

- **Embeddings:** Representaciones numéricas del texto que capturan el significado semántico.
- **FAISS:** Permite búsqueda eficiente entre miles de vectores en milisegundos.
- **Chunk overlap:** La superposición entre fragmentos garantiza que no se pierda contexto en los bordes.
- **Top-K:** Número de fragmentos más relevantes que se pasan a Claude como contexto.
