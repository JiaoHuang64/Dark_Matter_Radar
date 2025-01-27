# Dark Matter Radar

**Author:** Hengqixuan Yuan  
**Supervisor:** Dr. Yiming Zhong  

The **Dark Matter Radar** scrapes new articles from arXiv, processes them, ranks and selects images, and predicts their relevance using a pre-trained machine learning model.

## Features

- **Automated arXiv Scraping**: Fetches new articles from specified categories.
- **Relevance Prediction**: Uses a trained `Ridge` regression model to determine article relevance.
- **PDF & LaTeX Processing**: Extracts images and author information from PDFs and LaTeX source files.
- **Web Scraping**: Uses BeautifulSoup to extract metadata from arXiv pages.
- **Automated Data Saving**: Saves processed article data to CSV files.
- **Automated Execution**: Uses macOS Automator to schedule periodic execution of scripts.

## Installation

### 1. Clone the repository
```sh
git clone https://github.com/your-repo/Dark_Matter_Radar.git
cd Dark_Matter_Radar
```

### 2. Modify File Paths
Before running the scripts, update any occurrences or similar file paths of:
```python
os.chdir('/path/to/your/project/')
```
Replace `/path/to/your/project/` with the absolute path to your project directory and actual file paths.

### 3. Install dependencies
```sh
pip install -r Requirements.txt
```
For macOS users, ensure you have `poppler-utils` installed:
```sh
# macOS (via Homebrew)
brew install poppler
```

## Files & Directories

- `Scrape_Daily_New.py` - Main script for scraping and processing arXiv articles.
- `Mastodon_Post.py` - Script for posting filtered articles to Mastodon.
- `Wordpress_Post.py` - Script for posting filtered articles to WordPress.
- `Automation_Script` - AppleScript file used in Automator.
- `Requirements.txt` - Lists required Python packages.
- `relevance_model4.pkl` - Pre-trained model for relevance prediction.
- `processed_arxiv_ids.csv` - Stores already processed arXiv IDs.
- `filtered.csv` - Stores relevant articles after processing.
- `downloads/` - Directory where PDFs and LaTeX source files are stored.
- `viewable_images/` - Directory where extracted images are saved.

## Usage

### Running the Scraper
To fetch new articles and process them, run:
```sh
python Scrape_Daily_New.py
```
This script will:
- Fetch new articles from arXiv categories (`astro-ph`, `gr-qc`, `hep-ex`, `hep-ph`, `hep-th`).
- Extract metadata (title, abstract, authors, PDF link, etc.).
- Predict relevance scores using the trained `relevance_model4.pkl`.
- Save relevant articles to `filtered.csv`.
- Extract and save images from PDFs or LaTeX source files.

### Setting Up Automator for Scheduled Execution
To automate execution of the scripts:

1. **Open Automator** on macOS and create a new **Application**.
2. **Add "Run AppleScript" action**.
3. **Paste the `Automation_Script`**.
4. **Replace `/path/to/your/project/` with your actual project directory path**.
5. **Save the Automator Application**.
6. **Open macOS Calendar and create a new event**.
7. **Set it to repeat as desired daily based on your timezone**.
8. **In the event settings, select "Open file" and choose the Automator Application**.

Now, the scraper and posting scripts will run automatically at the scheduled times.

## Dependencies

Ensure all dependencies are installed via:
```sh
pip install -r Requirements.txt
```

### Key dependencies include:
- `numpy`, `pandas`, `scipy`, `matplotlib`
- `scikit-learn`, `joblib`
- `requests`, `beautifulsoup4`
- `pdf2image`, `Pillow`, `fitz` (PyMuPDF)
- `opencv-python`
- `arxiv` (API wrapper for arXiv)

## License

This project is licensed under the MIT License.

---

For questions or contributions, feel free to open an issue or submit a pull request on GitHub.
