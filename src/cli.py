"""CLI tool for ResearchMind AI Deep Research and Hybrid RAG platform."""

import argparse
import json
import sys
from backend.app.services.rag_engine import rag_engine
from backend.app.services.deep_researcher import deep_researcher
from backend.app.services.document_store import document_store


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="researchmind",
        description="ResearchMind AI: Production-Grade Deep Research & Hybrid RAG Platform",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. Search / RAG query
    search_parser = subparsers.add_parser("search", help="Execute hybrid RAG search across indexed papers")
    search_parser.add_argument("--query", "-q", type=str, required=True, help="Research inquiry or question")
    search_parser.add_argument("--top-k", "-k", type=int, default=6, help="Number of citations to retrieve")
    search_parser.add_argument("--format", "-f", type=str, default="text", choices=["text", "json"], help="Output format")

    # 2. Deep Research synthesis
    deep_parser = subparsers.add_parser("deep-research", help="Perform cross-document claim extraction and synthesis")
    deep_parser.add_argument("--topic", "-t", type=str, required=True, help="Scientific research topic or question")
    deep_parser.add_argument("--format", "-f", type=str, default="text", choices=["text", "json"], help="Output format")

    # 3. Server launch
    server_parser = subparsers.add_parser("server", help="Launch the FastAPI backend server")
    server_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    server_parser.add_argument("--port", "-p", type=int, default=8000, help="Port to listen on (default: 8000)")
    server_parser.add_argument("--reload", action="store_true", help="Enable automatic code reloading")

    return parser


def run_search(args: argparse.Namespace) -> int:
    res = rag_engine.query(query=args.query, top_k=args.top_k)

    if args.format == "json":
        print(res.model_dump_json(indent=2))
        return 0

    print("\n" + "=" * 76)
    print(f"🔬 RESEARCHMIND AI HYBRID RAG SYNTHESIS")
    print(f"❓ Query         : {res.query}")
    print(f"⏱️  Latency       : {res.latency_ms:.2f} ms | Chunks Retrieved: {res.retrieved_chunks_count}")
    print("=" * 76)
    print("\n--- GROUNDED SYNTHESIS ---")
    print(res.answer)
    print("\n--- CLAIM-LEVEL CITATIONS ---")
    for idx, c in enumerate(res.citations, 1):
        print(f"[{idx}] {c.doc_title} (Page {c.page_number}) [RRF Score: {c.score:.4f}]")
        print(f"    Snippet: \"{c.snippet.strip()}\"\n")
    print("=" * 76 + "\n")
    return 0


def run_deep_research(args: argparse.Namespace) -> int:
    res = deep_researcher.deep_research(topic=args.topic)

    if args.format == "json":
        print(res.model_dump_json(indent=2))
        return 0

    print("\n" + "=" * 76)
    print(f"🧠 RESEARCHMIND DEEP RESEARCH CROSS-DOCUMENT REASONING")
    print(f"🔬 Topic         : {res.topic}")
    print(f"⏱️  Latency       : {res.latency_ms:.2f} ms")
    print("=" * 76)
    print("\n--- EXECUTIVE SYNTHESIS ---")
    print(res.executive_synthesis)
    print("\n--- COMPARATIVE MATRIX ---")
    print(res.comparative_matrix_markdown)
    if res.contradictions:
        print("\n--- DETECTED CONTRADICTIONS & DIVERGENT CLAIMS ---")
        for c in res.contradictions:
            print(f"⚠️  {c.topic}:")
            print(f"    • {c.doc_a_title} (p. {c.doc_a_page}): {c.finding_a}")
            print(f"    • {c.doc_b_title} (p. {c.doc_b_page}): {c.finding_b}")
            print(f"    • Nature: {c.contradiction_nature}\n")
    print("=" * 76 + "\n")
    return 0


def run_server(args: argparse.Namespace) -> int:
    import uvicorn
    print(f"🚀 Launching ResearchMind AI Server on http://{args.host}:{args.port}")
    uvicorn.run("backend.app.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main():
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "search":
        code = run_search(args)
        sys.exit(code)
    elif args.command == "deep-research":
        code = run_deep_research(args)
        sys.exit(code)
    elif args.command == "server":
        code = run_server(args)
        sys.exit(code)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
