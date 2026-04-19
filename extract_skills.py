"""
extract_skills.py
=================
One-shot script: load the existing index, extract skills from all chunks,
and save the skills metadata store. Run this ONCE after indexing.

Usage:
    python extract_skills.py
    python extract_skills.py --model openai/gpt-4o-mini --batch 20
"""

import argparse
import os
import pickle
import sys
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Extract skills from indexed chunks")
    parser.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY"))
    parser.add_argument("--model",   default="openai/gpt-4o-mini")
    parser.add_argument("--batch",   type=int, default=10,
                        help="Chunks per batch before rate-limit pause")
    args = parser.parse_args()

    if not args.api_key:
        print("[ERROR] Set OPENROUTER_API_KEY in .env or pass --api-key")
        sys.exit(1)

    # Load existing chunks
    chunks_file = "vectorstore/chunks_metadata.pkl"
    if not os.path.exists(chunks_file):
        print(f"[ERROR] {chunks_file} not found. Run app.py and index documents first.")
        sys.exit(1)

    with open(chunks_file, "rb") as f:
        chunks = pickle.load(f)
    print(f"[INFO] Loaded {len(chunks)} chunks from {chunks_file}")

    # Extract skills
    from skills.extractor import SkillExtractor
    from skills.metadata_store import SkillsMetadataStore

    extractor = SkillExtractor(
        api_key=args.api_key,
        model=args.model,
        batch_size=args.batch,
    )

    print("[INFO] Extracting chunk-level skills...")
    enriched_chunks = extractor.extract_batch(chunks, show_progress=True)

    print("[INFO] Aggregating document-level profiles...")
    doc_profiles = extractor.aggregate_document_skills(enriched_chunks)

    print("[INFO] Saving skills metadata store...")
    store = SkillsMetadataStore()
    store.build_from_chunks(enriched_chunks, doc_profiles)

    # Also update the chunks pickle so skills_metadata is persisted
    with open(chunks_file, "wb") as f:
        pickle.dump(enriched_chunks, f)

    print("\n✅ Skills extraction complete!")
    print(f"   Total skills indexed: {len(store.get_all_skills())}")
    print(f"   Domains found: {store.get_all_domains()}")
    print(f"   Difficulty levels: {store.get_all_difficulties()}")
    print("\nYou can now filter by skill, domain, and difficulty in app_v2.py")


if __name__ == "__main__":
    main()
