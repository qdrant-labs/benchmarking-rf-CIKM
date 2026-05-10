"""
Step 2: Embed and index a BEIR corpus into Qdrant.

Reads beir_data_path.json (written by load_beir.py).
Collection name is derived as: {dataset}-relevance-feedback-{model-name}-docs

Example:
    python indexing_to_qdrant.py --dataset fiqa --model-name mxbread  # cloud inference
    python indexing_to_qdrant.py --dataset fiqa --model-name jina --no-cloud-inference  # fastembed (local)
"""

import argparse
import json
import os
from typing import Generator

from dotenv import load_dotenv

load_dotenv()

import tqdm
from beir.datasets.data_loader import GenericDataLoader
from qdrant_client import QdrantClient, models

# (model_id, vector_size, cloud_inference_default)
MODELS: dict[str, tuple[str, int, bool]] = {
    "jina":    ("jinaai/jina-embeddings-v2-base-en",    768, False),  # fastembed only
    "mxbread": ("mixedbread-ai/mxbai-embed-large-v1", 1024, True),   # cloud inference
}

DATA_PATH_FILE = "beir_data_path.json"


def vector_params(model_name: str) -> models.VectorParams:
    _, size, _ = MODELS[model_name]
    return models.VectorParams(size=size, distance=models.Distance.COSINE)


def point_generator(
    corpus: dict,
    model_name: str,
    model_id: str,
) -> Generator[models.PointStruct, None, None]:
    for idx, (doc_id, doc) in enumerate(corpus.items()):
        text = (doc.get("title") or "") + "\n" + (doc.get("text") or "")
        yield models.PointStruct(
            id=idx,
            vector={model_name: models.Document(text=text, model=model_id)},
            payload={"document": text, "document_id": doc_id},
        )


def main():
    parser = argparse.ArgumentParser(description="Index a BEIR corpus into Qdrant")
    parser.add_argument(
        "--dataset",
        default="scidocs",
        help="BEIR dataset name to index, must be present in beir_data_path.json (default: scidocs)",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        choices=list(MODELS.keys()),
        help="Embedding model to use",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--cloud-inference",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use Qdrant cloud inference for embeddings. Defaults to per-model setting (mxbread: True, jina: False).",
    )
    args = parser.parse_args()

    _, _, cloud_inference_default = MODELS[args.model_name]
    cloud_inference = args.cloud_inference if args.cloud_inference is not None else cloud_inference_default

    collection_name = f"{args.dataset}-relevance-feedback-{args.model_name}-docs"

    with open(DATA_PATH_FILE) as f:
        data_path = json.load(f)[args.dataset]

    corpus, _, _ = GenericDataLoader(data_folder=data_path).load(split="test")
    print(f"Loaded {len(corpus)} documents from '{args.dataset}'")

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, cloud_inference=cloud_inference)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={args.model_name: vector_params(args.model_name)},
        )
        print(f"Created collection '{collection_name}'")
    else:
        print(f"Collection '{collection_name}' already exists, skipping creation")

    model_id, _, _ = MODELS[args.model_name]
    print(f"Uploading {len(corpus)} documents (model: {model_id})...")
    client.upload_points(
        collection_name=collection_name,
        points=tqdm.tqdm(
            point_generator(corpus, args.model_name, model_id),
            total=len(corpus),
            unit="doc",
        ),
        batch_size=args.batch_size,
        parallel=args.parallel,
        max_retries=args.max_retries,
    )

    info = client.get_collection(collection_name)
    print(f"Done. Collection '{collection_name}' has {info.points_count} points.")


if __name__ == "__main__":
    main()
