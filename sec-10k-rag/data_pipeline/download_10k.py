"""
Download SEC 10-K filings - SIMPLIFIED VERSION THAT WORKS
Uses the SEC EDGAR Full Text Search to get documents
"""
import time
import requests
from typing import List, Optional, Dict
from pathlib import Path
import re

from config.config import config
from utils.logger import setup_logger

logger = setup_logger(__name__, config.LOGS_DIR / "download.log")


class SECDownloader:
    """Download SEC 10-K filings using EDGAR full-text search."""
    
    def __init__(self, tickers: Optional[Dict[str, str]] = None):
        self.tickers = tickers or config.TICKER_TO_CIK
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.USER_AGENT,
            'Accept-Encoding': 'gzip, deflate'
        })
        logger.info(f"Initialized downloader with {len(self.tickers)} tickers")
    
    def get_filing_urls(self, cik: str, ticker: str, num_filings: int = 3) -> List[tuple]:
        """
        Get 10-K filing URLs for a company.
        
        Returns:
            List of tuples: (accession_number, filing_date, document_url)
        """
        cik_padded = cik.zfill(10)
        
        # Get company filings metadata
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            recent = data.get('filings', {}).get('recent', {})
            forms = recent.get('form', [])
            accessions = recent.get('accessionNumber', [])
            dates = recent.get('filingDate', [])
            primary_docs = recent.get('primaryDocument', [])
            
            # Find 10-K filings
            filings = []
            for i, form in enumerate(forms):
                if form == '10-K' and len(filings) < num_filings:
                    accession = accessions[i]
                    date = dates[i]
                    primary_doc = primary_docs[i] if i < len(primary_docs) else None
                    
                    # Construct document URL
                    acc_no_dash = accession.replace('-', '')
                    cik_num = int(cik)
                    
                    if primary_doc:
                        doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{acc_no_dash}/{primary_doc}"
                    else:
                        # Fallback: try to get the full submission file
                        doc_url = f"https://www.sec.gov/cgi-bin/viewer?action=view&cik={cik_num}&accession_number={accession}&xbrl_type=v"
                    
                    filings.append((accession, date, doc_url))
            
            logger.info(f"Found {len(filings)} 10-K(s) for {ticker}")
            return filings
            
        except Exception as e:
            logger.error(f"Error fetching filings for {ticker}: {e}")
            return []
    
    def download_document(self, ticker: str, accession: str, date: str, url: str) -> Optional[Path]:
        """Download a single document."""
        time.sleep(config.RATE_LIMIT_DELAY)
        
        try:
            logger.info(f"Downloading {ticker} - {date}...")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Save document
            acc_clean = accession.replace('-', '')
            filename = f"{ticker}_{acc_clean}.txt"
            filepath = config.RAW_10K_DIR / filename
            
            # Save content (limit size)
            content = response.text[:config.MAX_FILE_SIZE]
            
            with open(filepath, 'w', encoding='utf-8', errors='ignore') as f:
                f.write(content)
            
            logger.info(f"✓ Downloaded {ticker} - {date} ({len(content)} chars)")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to download {ticker} - {date}: {e}")
            return None
    
    def download_all(self) -> List[Path]:
        """Download all 10-Ks for all companies."""
        logger.info(f"Starting download for {len(self.tickers)} companies")
        all_files = []
        
        for ticker, cik in self.tickers.items():
            try:
                print(f"\nProcessing {ticker}...")
                
                # Get filing URLs
                filings = self.get_filing_urls(cik, ticker)
                
                if not filings:
                    print(f"  ⚠ No filings found for {ticker}")
                    continue
                
                # Download each filing
                for accession, date, url in filings:
                    filepath = self.download_document(ticker, accession, date, url)
                    if filepath:
                        all_files.append(filepath)
                        print(f"  ✓ {ticker} - {date}")
                
                # Rate limiting between companies
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                print(f"  ✗ Error: {e}")
        
        return all_files


def main():
    """Main execution."""
    print("\n" + "="*60)
    print("SEC 10-K Document Downloader")
    print("="*60)
    
    # Check user agent
    if "Mozilla" in config.USER_AGENT or "example" in config.USER_AGENT:
        print("\n⚠ WARNING: Please set a proper User-Agent in .env file!")
        print("Format: YourName/1.0 (your.email@example.com)")
        print("\nContinuing anyway...\n")
    
    downloader = SECDownloader()
    
    try:
        files = downloader.download_all()
        
        print(f"\n{'='*60}")
        if files:
            print(f"✓ Successfully downloaded {len(files)} documents")
            print(f"Saved to: {config.RAW_10K_DIR.absolute()}")
        else:
            print("⚠ No files were downloaded")
            print("\nTroubleshooting:")
            print("1. Check internet connection")
            print("2. Set User-Agent in .env file")
            print("3. Check logs/download.log")
        print(f"{'='*60}\n")
        
    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}\n")
        raise


if __name__ == '__main__':
    main()