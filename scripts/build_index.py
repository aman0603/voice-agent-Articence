"""Build vector and BM25 indices from technical manuals."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from langchain_text_splitters import RecursiveCharacterTextSplitter
import pypdf

from src.config import get_settings
from src.retrieval import DenseSearcher, SparseSearcher

console = Console()


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF file."""
    try:
        reader = pypdf.PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n\n".join(text_parts)
    except Exception as e:
        console.print(f"[red]Error reading {pdf_path}: {e}[/red]")
        return ""


def load_documents(manuals_dir: Path) -> list[tuple[str, str]]:
    """Load all documents from manuals directory."""
    documents = []
    
    # Handle PDFs
    for pdf_path in manuals_dir.glob("*.pdf"):
        console.print(f"📄 Loading: {pdf_path.name}")
        text = extract_text_from_pdf(pdf_path)
        if text:
            documents.append((pdf_path.stem, text))
    
    # Handle text files
    for txt_path in manuals_dir.glob("*.txt"):
        console.print(f"📄 Loading: {txt_path.name}")
        text = txt_path.read_text(encoding='utf-8')
        if text:
            documents.append((txt_path.stem, text))
    
    # Handle markdown files
    for md_path in manuals_dir.glob("*.md"):
        console.print(f"📄 Loading: {md_path.name}")
        text = md_path.read_text(encoding='utf-8')
        if text:
            documents.append((md_path.stem, text))
    
    return documents


def chunk_documents(
    documents: list[tuple[str, str]], 
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> tuple[list[str], list[str]]:
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    all_chunks = []
    all_ids = []
    
    for doc_name, text in documents:
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{doc_name}:chunk_{i}")
    
    return all_chunks, all_ids


def main():
    """Build indices from documents."""
    console.print("[bold blue]Building Knowledge Base Indices[/bold blue]\n")
    
    settings = get_settings()
    
    # Create directories
    script_dir = Path(__file__).parent.parent
    manuals_dir = script_dir / settings.manuals_dir
    index_dir = script_dir / settings.index_dir
    
    if not manuals_dir.exists():
        console.print(f"[red]Manuals directory not found: {manuals_dir}[/red]")
        console.print("Run [cyan]python scripts/download_manuals.py[/cyan] first.")
        return
    
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Load documents
    console.print("[bold]Step 1: Loading documents[/bold]")
    documents = load_documents(manuals_dir)
    
    if not documents:
        console.print("[red]No documents found![/red]")
        return
    
    console.print(f"Loaded {len(documents)} documents\n")
    
    # Chunk documents
    console.print("[bold]Step 2: Chunking documents[/bold]")
    chunks, chunk_ids = chunk_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    console.print(f"Created {len(chunks)} chunks\n")
    
    # Build dense index
    console.print("[bold]Step 3: Building dense vector index[/bold]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        progress.add_task("Encoding embeddings...", total=None)
        
        dense_searcher = DenseSearcher(index_path=str(index_dir))
        dense_searcher.build_index(chunks, chunk_ids)
    
    console.print(f"[green]Dense index saved to {index_dir / 'dense.index'}[/green]\n")
    
    # Build sparse index
    console.print("[bold]Step 4: Building BM25 index[/bold]")
    sparse_searcher = SparseSearcher(index_path=str(index_dir))
    sparse_searcher.build_index(chunks, chunk_ids)
    console.print(f"[green]BM25 index saved to {index_dir / 'bm25.pkl'}[/green]\n")
    
    # Summary
    console.print("[bold green]Index building complete![/bold green]")
    console.print(f"""
Summary:
- Documents: {len(documents)}
- Chunks: {len(chunks)}
- Average chunk size: {sum(len(c) for c in chunks) // len(chunks)} chars
- Dense index: {index_dir / 'dense.index'}
- Sparse index: {index_dir / 'bm25.pkl'}

Run [cyan]python -m src.main[/cyan] to start the server.
""")


if __name__ == "__main__":
    main()
