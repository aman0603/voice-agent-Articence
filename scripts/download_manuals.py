"""Download sample Dell PowerEdge manuals for the knowledge base."""

import os
import asyncio
from pathlib import Path
import httpx
from rich.console import Console
from rich.progress import Progress

console = Console()

# Sample Dell PowerEdge documentation URLs (public PDFs)
# These are placeholder URLs - replace with actual Dell documentation
SAMPLE_DOCS = [
    {
        "name": "Dell PowerEdge R750 Installation Guide",
        "filename": "dell_r750_installation.pdf",
        # Using a sample technical PDF for demo
        "url": "https://downloads.dell.com/manuals/all-products/esuprt_ser_stor_net/esuprt_poweredge/poweredge-r750_owners-manual_en-us.pdf"
    },
    {
        "name": "Dell PowerEdge R640 Troubleshooting Guide", 
        "filename": "dell_r640_troubleshooting.pdf",
        "url": "https://downloads.dell.com/manuals/all-products/esuprt_ser_stor_net/esuprt_poweredge/poweredge-r640_owners-manual_en-us.pdf"
    },
]


async def download_file(client: httpx.AsyncClient, url: str, path: Path) -> bool:
    """Download a file with progress."""
    try:
        async with client.stream('GET', url, follow_redirects=True) as response:
            if response.status_code != 200:
                console.print(f"[red]Failed to download: {response.status_code}[/red]")
                return False
            
            total = int(response.headers.get('content-length', 0))
            
            with open(path, 'wb') as f:
                downloaded = 0
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
            
            return True
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return False


async def main():
    """Download all sample documents."""
    # Create directories
    script_dir = Path(__file__).parent.parent
    manuals_dir = script_dir / "data" / "manuals"
    manuals_dir.mkdir(parents=True, exist_ok=True)
    
    console.print("[bold blue]Downloading Dell PowerEdge Documentation[/bold blue]")
    console.print(f"Target directory: {manuals_dir}\n")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        with Progress() as progress:
            task = progress.add_task("[green]Downloading...", total=len(SAMPLE_DOCS))
            
            for doc in SAMPLE_DOCS:
                console.print(f"📄 {doc['name']}")
                path = manuals_dir / doc['filename']
                
                if path.exists():
                    console.print(f"   [yellow]Already exists, skipping[/yellow]")
                    progress.advance(task)
                    continue
                
                success = await download_file(client, doc['url'], path)
                
                if success:
                    size_mb = path.stat().st_size / (1024 * 1024)
                    console.print(f"   [green]Downloaded ({size_mb:.1f} MB)[/green]")
                else:
                    console.print(f"   [red]Failed - creating sample text file instead[/red]")
                    # Create a sample text file for testing
                    sample_path = path.with_suffix('.txt')
                    sample_path.write_text(create_sample_content(doc['name']))
                    console.print(f"   [yellow]Created sample: {sample_path.name}[/yellow]")
                
                progress.advance(task)
    
    console.print("\n[bold green]Download complete![/bold green]")
    console.print(f"Run [cyan]python scripts/build_index.py[/cyan] to index the documents.")


def create_sample_content(title: str) -> str:
    """Create sample technical content for testing."""
    return f"""# {title}

## Chapter 1: Power Supply

### 1.1 Power Redundancy Modes

The PowerEdge server supports the following power redundancy modes:

1. **Grid Redundancy Mode**: Power supplies are divided into two grids. If one grid fails, the other grid provides power to the system.

2. **Power Supply Redundancy Mode**: If one power supply fails, the remaining power supplies continue to provide power to the system.

3. **No Redundancy Mode**: All installed power supplies share the power load. The system does not have power redundancy.

### 1.2 Power Supply LED Indicators

| LED Color | Behavior | Condition |
|-----------|----------|-----------|
| Green | On | Power supply is operational |
| Green | Blinking | Hot-spare mode active |
| Amber | On | Power supply fault |
| Amber | Blinking | Power supply warning |
| Off | - | No AC power or PSU not installed |

### 1.3 Troubleshooting Power Issues

If the power supply LED shows amber:
1. Check the AC power cord connection
2. Verify the outlet is functioning
3. Ensure the power supply is fully seated
4. Check for debris in the power supply bay

## Chapter 2: Storage Configuration

### 2.1 RAID Levels

The server supports the following RAID configurations:

- **RAID 0**: Striping, no redundancy, maximum performance
- **RAID 1**: Mirroring, single-drive fault tolerance
- **RAID 5**: Striping with parity, single-drive fault tolerance
- **RAID 6**: Striping with dual parity, two-drive fault tolerance
- **RAID 10**: Mirrored stripes, balanced performance and redundancy

### 2.2 Drive LED Indicators

Each drive bay has two LEDs:
- Activity LED (green): Indicates drive activity
- Status LED (green/amber): Indicates drive health

Blinking patterns:
- Green blinking: Normal activity
- Amber blinking 1x/sec: Drive warning
- Solid amber: Drive failure

## Chapter 3: Network Configuration

### 3.1 Network Port Speeds

- 1 Gbps: Standard Ethernet
- 10 Gbps: SFP+ or 10GBASE-T
- 25 Gbps: SFP28

### 3.2 Troubleshooting Network Issues

1. Check link LED on the NIC
2. Verify cable connection
3. Check switch port status
4. Run network diagnostics from iDRAC

## Chapter 4: Memory Configuration

### 4.1 DIMM Population Rules

- Populate DIMMs in order: A1, A2, A3, A4
- Match DIMM speeds within a channel
- Use ECC memory only
- Maximum 32 DIMMs supported

### 4.2 Memory Error Handling

The system logs memory errors to the System Event Log:
- Correctable ECC errors: Logged only
- Uncorrectable ECC errors: System halt

## Chapter 5: BIOS and Boot

### 5.1 Boot Sequence

1. Power on
2. POST begins
3. BIOS initialization
4. Boot device selection
5. OS load

### 5.2 Common POST Error Codes

- E1000: CPU failure
- E2000: Memory failure  
- E3000: Storage controller failure
- E4000: NIC failure
"""


if __name__ == "__main__":
    asyncio.run(main())
