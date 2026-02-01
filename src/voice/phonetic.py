"""Phonetic converter for technical term pronunciation."""

import re
from typing import Dict


# Technical term to spoken form mappings
PHONETIC_MAP: Dict[str, str] = {
    # Storage
    "RAID": "raid",
    "RAID0": "raid zero",
    "RAID1": "raid one", 
    "RAID5": "raid five",
    "RAID6": "raid six",
    "RAID10": "raid ten",
    "SSD": "S-S-D",
    "HDD": "H-D-D",
    "NVMe": "N-V-M-E",
    "SAS": "sass",
    "SATA": "say-ta",
    
    # Network
    "SFP": "S-F-P",
    "SFP+": "S-F-P plus",
    "QSFP": "Q-S-F-P",
    "NIC": "nick",
    "VLAN": "V-lan",
    "iSCSI": "eye-scuzzy",
    "FC": "fiber channel",
    "Gbps": "gigabits per second",
    "Mbps": "megabits per second",
    
    # Power
    "PSU": "P-S-U",
    "UPS": "U-P-S",
    "AC": "A-C",
    "DC": "D-C",
    
    # Memory & CPU
    "DIMM": "dim",
    "RDIMM": "R-dim",
    "LRDIMM": "L-R-dim",
    "ECC": "E-C-C",
    "DDR4": "D-D-R four",
    "DDR5": "D-D-R five",
    "CPU": "C-P-U",
    "vCPU": "virtual C-P-U",
    
    # General
    "LED": "L-E-D",
    "USB": "U-S-B",
    "BIOS": "bye-ose",
    "UEFI": "you-eff-ee",
    "POST": "post",
    "iDRAC": "eye-drac",
    "BMC": "B-M-C",
    "IPMI": "I-P-M-I",
    "KVM": "K-V-M",
    
    # Units
    "GB": "gigabytes",
    "TB": "terabytes",
    "MB": "megabytes",
    "GHz": "gigahertz",
    "MHz": "megahertz",
    "KB": "kilobytes",
    "kW": "kilowatts",
    "BTU": "B-T-U",
    
    # Interface
    "PCIe": "P-C-I express",
    "PCI": "P-C-I",
    "SCSI": "scuzzy",
    "I/O": "I-O",
    "GPIO": "G-P-I-O",
}


class PhoneticConverter:
    """
    Converts technical terms to their spoken pronunciations.
    
    Ensures TTS produces natural-sounding technical content.
    """
    
    def __init__(self, custom_mappings: Dict[str, str] = None):
        """Initialize converter with optional custom mappings."""
        self.mappings = PHONETIC_MAP.copy()
        if custom_mappings:
            self.mappings.update(custom_mappings)
        
        # Pre-compile regex patterns for efficiency
        # Sort by length (longest first) to match "SFP+" before "SFP" and "RAID10" before "RAID"
        sorted_terms = sorted(self.mappings.keys(), key=len, reverse=True)
        
        # Create pattern that matches terms
        patterns = []
        for term in sorted_terms:
            # Escape special regex chars
            escaped = re.escape(term)
            # Use word boundary on left, but lookahead for non-word on right
            # This handles SFP+ followed by space or punctuation
            patterns.append(f"(?<![A-Za-z]){escaped}(?![A-Za-z])")
        
        self._pattern = re.compile(
            '(' + '|'.join(patterns) + ')',
            re.IGNORECASE
        )
    
    def convert(self, text: str) -> str:
        """
        Convert technical terms in text to phonetic forms.
        
        Args:
            text: Input text with technical terms
            
        Returns:
            Text with terms converted to spoken forms
        """
        def replace_match(match):
            term = match.group(1)
            # Try exact match first, then uppercase
            if term in self.mappings:
                return self.mappings[term]
            elif term.upper() in self.mappings:
                return self.mappings[term.upper()]
            return term
        
        return self._pattern.sub(replace_match, text)
    
    def add_mapping(self, term: str, pronunciation: str) -> None:
        """Add a custom term mapping."""
        self.mappings[term] = pronunciation
        # Rebuild pattern
        self.__init__(self.mappings)
    
    def format_number_sequences(self, text: str) -> str:
        """
        Format number sequences for better pronunciation.
        
        E.g., "192.168.1.1" -> "192 dot 168 dot 1 dot 1"
        """
        # IP addresses
        ip_pattern = r'\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b'
        text = re.sub(ip_pattern, r'\1 dot \2 dot \3 dot \4', text)
        
        # MAC addresses (xx:xx:xx:xx:xx:xx)
        mac_pattern = r'\b([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2})\b'
        text = re.sub(mac_pattern, r'\1 \2 \3 \4 \5 \6', text)
        
        return text
