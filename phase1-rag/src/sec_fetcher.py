import requests
import os
from html.parser import HTMLParser

HEADERS = {
    "User-Agent": "financial-research-agent karthikeyansetti@gmail.com"
}


def get_cik(ticker: str) -> str:
    """
    Converts a stock ticker to SEC CIK number.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=HEADERS)
    data = response.json()

    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry["ticker"] == ticker_upper:
            cik = str(entry["cik_str"]).zfill(10)
            return cik

    raise ValueError(f"Ticker {ticker} not found in SEC database")


def get_latest_10q_accession(cik: str) -> tuple:
    """
    Gets the accession number and date of the latest 10-Q filing.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS)
    data = response.json()

    filings = data["filings"]["recent"]
    forms = filings["form"]
    dates = filings["filingDate"]
    accession_numbers = filings["accessionNumber"]

    for i, form in enumerate(forms):
        if form == "10-Q":
            return accession_numbers[i], dates[i]

    raise ValueError(f"No 10-Q found for CIK {cik}")


def get_main_document_url(cik_int: int, accession_dashed: str) -> str:
    """
    Constructs the direct URL to the main 10-Q HTM document.
    First tries to find a direct link, then falls back to
    extracting the filename from the ix?doc= viewer link.
    """
    accession_nodash = accession_dashed.replace("-", "")
    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{accession_nodash}/{accession_dashed}-index.htm"
    )
    response = requests.get(index_url, headers=HEADERS)

    # first pass — look for direct HTM links that aren't exhibits
    class DirectLinkFinder(HTMLParser):
        def __init__(self):
            super().__init__()
            self.main_doc = None

        def handle_starttag(self, tag, attrs):
            if tag == "a" and not self.main_doc:
                for attr, val in attrs:
                    if (attr == "href" and val and
                            "/Archives/edgar/data" in val and
                            val.endswith(".htm") and
                            "ix?doc=" not in val and
                            "ex" not in val.split("/")[-1].lower()):
                        self.main_doc = val

    direct_parser = DirectLinkFinder()
    direct_parser.feed(response.text)

    if direct_parser.main_doc:
        return f"https://www.sec.gov{direct_parser.main_doc}"

    # fallback — extract filename from ix?doc= viewer link
    class IxLinkFinder(HTMLParser):
        def __init__(self):
            super().__init__()
            self.doc_name = None

        def handle_starttag(self, tag, attrs):
            if tag == "a" and not self.doc_name:
                for attr, val in attrs:
                    if (attr == "href" and val and
                            "ix?doc=" in val and
                            val.endswith(".htm")):
                        self.doc_name = val.split("/")[-1]

    ix_parser = IxLinkFinder()
    ix_parser.feed(response.text)

    if ix_parser.doc_name:
        return (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_int}/{accession_nodash}/{ix_parser.doc_name}"
        )

    raise ValueError("Could not find main document URL")


def download_10q(ticker: str, save_dir: str = "phase1-rag/data") -> str:
    """
    Takes a ticker, finds the latest 10-Q on SEC EDGAR,
    downloads the human-readable HTM document.
    Returns the path to the saved file.
    """
    print(f"Looking up CIK for {ticker}...")
    cik = get_cik(ticker)
    cik_int = int(cik)
    print(f"CIK: {cik}")

    print(f"Finding latest 10-Q...")
    accession_dashed, filing_date = get_latest_10q_accession(cik)
    print(f"Filing date: {filing_date}, accession: {accession_dashed}")

    print(f"Finding main document...")
    doc_url = get_main_document_url(cik_int, accession_dashed)
    print(f"Document URL: {doc_url}")

    response = requests.get(doc_url, headers=HEADERS)
    if response.status_code != 200:
        raise ValueError(f"Failed to download: {response.status_code}")

    os.makedirs(save_dir, exist_ok=True)
    filename = f"{ticker.lower()}_{filing_date}_10q.html"
    filepath = os.path.join(save_dir, filename)

    with open(filepath, "wb") as f:
        f.write(response.content)

    print(f"Saved to {filepath}")
    return filepath


if __name__ == "__main__":
    path = download_10q("MSFT")
    print(f"Done: {path}")