"""
Parse and clean SEC 10-K HTML/SGML documents.
"""
import re
import json
from pathlib import Path
from typing import Dict, Optional
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.config import config
from utils.logger import setup_logger
from utils.exceptions import ParsingError

logger = setup_logger(__name__, config.LOGS_DIR / "parse.log")


class DocumentParser:
    """
    Parse SEC 10-K documents from raw HTML/SGML to clean text.
    
    Handles:
    - HTML/SGML parsing
    - Section extraction (Item 1, Item 1A, etc.)
    - Text cleaning and normalization
    """
    
    # Common 10-K sections
    SECTION_PATTERNS = [
        (r'Item\s+1[^A-Za-z0-9]', 'Item 1 - Business'),
        (r'Item\s+1A', 'Item 1A - Risk Factors'),
        (r'Item\s+1B', 'Item 1B - Unresolved Staff Comments'),
        (r'Item\s+2', 'Item 2 - Properties'),
        (r'Item\s+3', 'Item 3 - Legal Proceedings'),
        (r'Item\s+4', 'Item 4 - Mine Safety Disclosures'),
        (r'Item\s+5', 'Item 5 - Market for Stock'),
        (r'Item\s+6', 'Item 6 - Selected Financial Data'),
        (r'Item\s+7[^A-Za-z0-9]', 'Item 7 - MD&A'),
        (r'Item\s+7A', 'Item 7A - Market Risk'),
        (r'Item\s+8', 'Item 8 - Financial Statements'),
        (r'Item\s+9[^A-Za-z0-9]', 'Item 9 - Accounting Disagreements'),
        (r'Item\s+9A', 'Item 9A - Controls and Procedures'),
        (r'Item\s+9B', 'Item 9B - Other Information'),
        (r'Item\s+10', 'Item 10 - Directors and Officers'),
        (r'Item\s+11', 'Item 11 - Executive Compensation'),
        (r'Item\s+12', 'Item 12 - Security Ownership'),
        (r'Item\s+13', 'Item 13 - Related Transactions'),
        (r'Item\s+14', 'Item 14 - Principal Accountant'),
        (r'Item\s+15', 'Item 15 - Exhibits'),
    ]
    
    def __init__(self):
        """Initialize parser."""
        self.processed_count = 0
        self.error_count = 0
    
    def html_to_text(self, content: str) -> str:
        """
        Convert HTML/SGML to clean text.
        
        Args:
            content: Raw HTML/SGML content
            
        Returns:
            Clean text string
        """
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script, style, and other non-content tags
            for tag in soup(['script', 'style', 'noscript', 'head', 'meta']):
                tag.decompose()
            
            # Extract text
            text = soup.get_text(separator='\n')
            
            # Clean up whitespace
            text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
            text = re.sub(r'[ \t]+', ' ', text)     # Normalize spaces
            text = re.sub(r' \n', '\n', text)       # Remove trailing spaces
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error converting HTML to text: {e}")
            raise ParsingError(f"HTML parsing failed: {e}") from e
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract standard 10-K sections from text.
        
        Args:
            text: Full document text
            
        Returns:
            Dictionary mapping section names to content
        """
        sections = {}
        
        # Find all section boundaries
        boundaries = []
        for pattern, name in self.SECTION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                boundaries.append((match.start(), name))
        
        # Sort by position
        boundaries.sort(key=lambda x: x[0])
        
        # Extract text between boundaries
        for i, (start_pos, section_name) in enumerate(boundaries):
            # End position is start of next section or end of document
            end_pos = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
            
            section_text = text[start_pos:end_pos].strip()
            
            # Only keep substantial sections (> 100 chars)
            if len(section_text) > 100:
                sections[section_name] = section_text
        
        # If no sections found, return full text
        if not sections:
            sections['FULL_TEXT'] = text
        
        return sections
    
    def parse_file(self, filepath: Path) -> Optional[Dict]:
        """
        Parse a single 10-K file.
        
        Args:
            filepath: Path to raw 10-K file
            
        Returns:
            Dictionary with parsed data, or None if failed
        """
        try:
            logger.info(f"Parsing {filepath.name}")
            
            # Read raw content
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                raw_content = f.read()
            
            # Convert to clean text
            clean_text = self.html_to_text(raw_content)
            
            # Extract sections
            sections = self.extract_sections(clean_text)
            
            # Extract metadata from filename
            filename = filepath.stem
            parts = filename.split('_')
            ticker = parts[0] if parts else 'UNKNOWN'
            
            # Prepare output
            result = {
                'ticker': ticker,
                'filename': filepath.name,
                'full_text': clean_text,
                'sections': sections,
                'word_count': len(clean_text.split()),
                'section_count': len(sections)
            }
            
            # Save clean text
            clean_filepath = config.CLEAN_TXT_DIR / f"{filename}.txt"
            with open(clean_filepath, 'w', encoding='utf-8') as f:
                f.write(clean_text)
            
            # Save sections as JSON
            sections_filepath = config.CLEAN_TXT_DIR / f"{filename}_sections.json"
            with open(sections_filepath, 'w', encoding='utf-8') as f:
                json.dump(sections, f, indent=2)
            
            logger.info(
                f"Parsed {ticker}: {result['word_count']} words, "
                f"{result['section_count']} sections"
            )
            
            self.processed_count += 1
            return result
            
        except Exception as e:
            logger.error(f"Error parsing {filepath.name}: {e}", exc_info=True)
            self.error_count += 1
            return None
    
    def parse_all(self, parallel: bool = True) -> Dict[str, Dict]:
        """
        Parse all raw 10-K files.
        
        Args:
            parallel: Use parallel processing
            
        Returns:
            Dictionary mapping filenames to parsed data
        """
        files = list(config.RAW_10K_DIR.glob('*.txt'))
        logger.info(f"Found {len(files)} files to parse")
        
        results = {}
        
        if parallel:
            with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
                futures = {
                    executor.submit(self.parse_file, f): f.name 
                    for f in files
                }
                
                for future in as_completed(futures):
                    filename = futures[future]
                    try:
                        result = future.result()
                        if result:
                            results[filename] = result
                    except Exception as e:
                        logger.error(f"Error processing {filename}: {e}")
        else:
            for filepath in files:
                result = self.parse_file(filepath)
                if result:
                    results[filepath.name] = result
        
        logger.info(
            f"Parsing complete: {self.processed_count} successful, "
            f"{self.error_count} errors"
        )
        
        return results


def main():
    """Main execution function."""
    parser = DocumentParser()
    
    try:
        results = parser.parse_all(parallel=True)
        
        # Summary statistics
        total_words = sum(r['word_count'] for r in results.values())
        total_sections = sum(r['section_count'] for r in results.values())
        
        print(f"\n{'='*60}")
        print(f"✓ Parsed {len(results)} documents")
        print(f"Total words: {total_words:,}")
        print(f"Total sections: {total_sections}")
        print(f"Output directory: {config.CLEAN_TXT_DIR.absolute()}")
        print(f"{'='*60}\n")
        
        # Save summary
        summary_path = config.CLEAN_TXT_DIR / "_parsing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'total_files': len(results),
                'total_words': total_words,
                'total_sections': total_sections,
                'files': list(results.keys())
            }, f, indent=2)
        
    except Exception as e:
        logger.error(f"Parsing failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()