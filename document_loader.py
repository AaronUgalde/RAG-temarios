"""
===============================================================================
MÓDULO: document_loader.py
PROPÓSITO: Carga, extracción y fragmentación de documentos PDF (Temarios)

Este módulo es responsable de:
1. Leer archivos PDF de la carpeta 'temarios/'
2. Extraer el texto de cada página
3. Dividir el texto en fragmentos manejables (chunks)
4. Asociar metadata a cada fragmento (carrera, nombre de archivo, página)

CONCEPTO RAG - PASO 1 (INDEXACIÓN):
En un sistema RAG (Retrieval-Augmented Generation), primero se deben
"indexar" los documentos. Esto significa prepararlos para búsqueda:
   Documento PDF → Texto → Fragmentos → Embeddings → Base vectorial

Este módulo cubre la parte: Documento PDF → Texto → Fragmentos
===============================================================================
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any

from pypdf import PdfReader


# =============================================================================
# CLASE PRINCIPAL: DocumentLoader
# =============================================================================

class DocumentLoader:
    """
    Carga y procesa documentos PDF para el sistema RAG.
    
    Atributos:
        temarios_path (Path): Ruta a la carpeta con los archivos PDF.
        chunk_size (int):     Tamaño máximo de cada fragmento en caracteres.
        chunk_overlap (int):  Caracteres de superposición entre fragmentos
                              contiguos para preservar el contexto.
    """

    def __init__(self, temarios_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Inicializa el cargador de documentos.

        Args:
            temarios_path: Ruta a la carpeta que contiene los PDFs.
            chunk_size:    Número de caracteres por fragmento.
            chunk_overlap: Número de caracteres compartidos entre fragmentos.
        """
        self.temarios_path = Path(temarios_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # -------------------------------------------------------------------------
    # MÉTODO PÚBLICO: cargar todos los documentos
    # -------------------------------------------------------------------------

    def load_all_documents(self) -> List[Dict[str, Any]]:
        """
        Carga y fragmenta todos los PDFs en la carpeta 'temarios/'.

        Returns:
            Lista de diccionarios. Cada diccionario representa un fragmento
            de texto con la siguiente estructura:
            {
                "text":     str,   # El texto del fragmento
                "source":   str,   # Nombre del archivo PDF
                "carrera":  str,   # Nombre inferido de la carrera
                "page":     int,   # Número de página de origen
                "chunk_id": int    # Índice del fragmento en el documento
            }
        """
        all_chunks = []

        # Obtener la lista de archivos PDF en la carpeta
        pdf_files = list(self.temarios_path.glob("*.pdf"))

        if not pdf_files:
            print(f"[ADVERTENCIA] No se encontraron PDFs en: {self.temarios_path}")
            return []

        for pdf_path in pdf_files:
            print(f"[INFO] Procesando: {pdf_path.name}")
            try:
                # Extraer texto del PDF y generar fragmentos
                chunks = self._process_pdf(pdf_path)
                all_chunks.extend(chunks)
                print(f"  → {len(chunks)} fragmentos generados.")
            except Exception as e:
                print(f"  [ERROR] No se pudo procesar {pdf_path.name}: {e}")

        print(f"\n[INFO] Total de fragmentos generados: {len(all_chunks)}")
        return all_chunks

    # -------------------------------------------------------------------------
    # MÉTODOS PRIVADOS: procesamiento interno
    # -------------------------------------------------------------------------

    def _process_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extrae el texto de un PDF y lo divide en fragmentos.

        Args:
            pdf_path: Ruta completa al archivo PDF.

        Returns:
            Lista de fragmentos con su metadata asociada.
        """
        reader = PdfReader(str(pdf_path))

        # Inferir el nombre de la carrera desde el nombre del archivo
        # Ejemplo: "temario_sistemas_computacionales.pdf" → "sistemas computacionales"
        carrera = self._infer_carrera_name(pdf_path.stem)

        all_chunks = []
        chunk_id = 0

        # Iterar página por página del PDF
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()

            # Saltar páginas vacías o con muy poco contenido
            if not page_text or len(page_text.strip()) < 50:
                continue

            # Limpiar el texto de caracteres problemáticos
            clean_text = self._clean_text(page_text)

            # Dividir el texto de la página en fragmentos (chunks)
            page_chunks = self._split_into_chunks(clean_text)

            for chunk_text in page_chunks:
                all_chunks.append({
                    "text":     chunk_text,
                    "source":   pdf_path.name,
                    "carrera":  carrera,
                    "page":     page_num + 1,   # Páginas comienzan en 1 para el usuario
                    "chunk_id": chunk_id
                })
                chunk_id += 1

        return all_chunks

    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Divide un texto largo en fragmentos de tamaño controlado.

        ESTRATEGIA DE FRAGMENTACIÓN:
        Se utiliza una ventana deslizante con superposición (overlap).
        Esto garantiza que el contexto no se pierda en los bordes de cada
        fragmento. Por ejemplo, con chunk_size=1000 y chunk_overlap=200:
          - Fragmento 1: caracteres [0, 1000]
          - Fragmento 2: caracteres [800, 1800]   ← 200 de overlap
          - Fragmento 3: caracteres [1600, 2600]
        
        Args:
            text: Texto completo a fragmentar.

        Returns:
            Lista de cadenas de texto (fragmentos).
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size

            # Intentar cortar en un espacio o salto de línea para no romper palabras
            if end < text_length:
                # Buscar el último espacio dentro del rango para corte limpio
                cut_point = text.rfind(" ", start, end)
                if cut_point != -1:
                    end = cut_point

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Avanzar restando el overlap para mantener contexto
            start = end - self.chunk_overlap

        return chunks

    def _clean_text(self, text: str) -> str:
        """
        Limpia y normaliza el texto extraído del PDF.

        Args:
            text: Texto crudo del PDF.

        Returns:
            Texto limpio y normalizado.
        """
        # Reemplazar múltiples saltos de línea por uno solo
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Reemplazar múltiples espacios por uno solo
        text = re.sub(r' {2,}', ' ', text)
        # Eliminar caracteres de control no imprimibles
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return text.strip()

    def _infer_carrera_name(self, filename_stem: str) -> str:
        """
        Infiere el nombre legible de una carrera a partir del nombre del archivo.
        
        Ejemplo: "temario_ing_sistemas" → "ing sistemas"
        
        Args:
            filename_stem: Nombre del archivo sin extensión.

        Returns:
            Nombre formateado de la carrera.
        """
        # Reemplazar guiones y guiones bajos por espacios
        name = filename_stem.replace("_", " ").replace("-", " ")
        # Eliminar prefijos comunes como "temario", "plan", "programa"
        name = re.sub(r'^(temario|plan|programa|pensum)\s+', '', name, flags=re.IGNORECASE)
        return name.strip().title()

    # -------------------------------------------------------------------------
    # MÉTODO UTILITARIO: listar carreras disponibles
    # -------------------------------------------------------------------------

    def get_available_careers(self) -> List[str]:
        """
        Retorna la lista de nombres de carreras disponibles en la carpeta.

        Returns:
            Lista de nombres de carrera inferidos de los archivos PDF.
        """
        pdf_files = list(self.temarios_path.glob("*.pdf"))
        return [self._infer_carrera_name(f.stem) for f in pdf_files]
