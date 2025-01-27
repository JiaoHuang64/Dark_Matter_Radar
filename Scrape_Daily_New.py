import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import ssl
import re
import shutil
import tarfile
from collections import Counter
import random
from datetime import datetime, timedelta
import hashlib
import fitz  # PyMuPDF
from PIL import Image
import cv2
import string
import joblib
from sklearn.feature_extraction.text import HashingVectorizer
import arxiv  # arxiv API
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge


# Define WeightedRidge classes
class WeightedRidge(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        print(f"Initializing WeightedRidge with alpha: {alpha}")
        self.model = Ridge(alpha=self.alpha)

    def fit(self, X, y, sample_weight=None):
        print("Fitting the model...")
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        print("Making predictions...")
        return self.model.predict(X)

    def set_params(self, **params):
        if 'alpha' in params:
            self.alpha = params['alpha']
            self.model.set_params(alpha=self.alpha)
        return self

    def get_params(self, deep=True):
        return {'alpha': self.alpha}


# Scrape_Daily_New.py
print("Starting Scrape_Daily_New.py")
os.chdir('/Users/jiaohuangbixia/Downloads/1/MODELTRAIN')
# Ignore SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

PROCESSED_IDS_FILE = 'processed_arxiv_ids.csv'
PROCESSED_IDS_REPORT = 'processed_arxiv_ids_report.html'


def create_directory(path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def load_processed_ids():
    """Load processed arxiv IDs from a CSV file."""
    if os.path.exists(PROCESSED_IDS_FILE):
        df = pd.read_csv(PROCESSED_IDS_FILE)
        return set(df['arxiv_id'].tolist())
    return set()


def save_processed_ids(processed_ids):
    """Save processed arxiv IDs to a CSV file."""
    df = pd.DataFrame(list(processed_ids), columns=['arxiv_id'])
    df.to_csv(PROCESSED_IDS_FILE, index=False)
    generate_processed_ids_report(df)


def generate_processed_ids_report(df):
    """Generate an HTML report of processed arxiv IDs."""
    html = df.to_html(index=False)
    with open(PROCESSED_IDS_REPORT, 'w') as f:
        f.write(html)


def fetch_arxiv_new_articles(category):
    """Fetch new articles from arxiv for a given category."""
    url = f'https://arxiv.org/list/{category}/new'
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch data from {url}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    articles = []
    for dt in soup.find_all('dt'):
        a = dt.find('a', title='Abstract')
        if a:
            arxiv_id = a.get_text().strip().replace('\n', '').replace('arXiv:', '').replace(' ', '')
            articles.append(arxiv_id)
    return articles


def fetch_article_details(arxiv_id):
    """Fetch the details of an arxiv article by ID."""
    url = f'https://arxiv.org/abs/{arxiv_id}'
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch details for {arxiv_id}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.find('h1', class_='title mathjax').get_text(strip=True).replace('Title:', '').strip()
    abstract = soup.find('blockquote', class_='abstract mathjax').get_text(strip=True).replace('Abstract:', '').strip()
    return {
        'title': title,
        'abstract': abstract,
        'arxiv_id': arxiv_id,
        'pdf_url': f'https://arxiv.org/pdf/{arxiv_id}.pdf',
        'html_url': f'https://arxiv.org/abs/{arxiv_id}'
    }


def fetch_author_info_arxiv_api(arxiv_id):
    """Fetch the authors of an arxiv article by ID using direct API request."""
    try:
        search = arxiv.Search(id_list=[arxiv_id])
        for result in search.results():
            return [str(author) for author in result.authors]
    except Exception as e:
        print(f"Error fetching authors via API for {arxiv_id}: {e}")
    return None


def download_tex_from_arxiv(arxiv_id, download_dir='downloads'):
    """Download the LaTeX source file of an arxiv article by ID."""
    source_url = f"https://arxiv.org/e-print/{arxiv_id}"
    response = requests.get(source_url)
    if response.status_code == 200:
        content_type = response.headers.get('content-type')
        print(f"Downloaded content type for {arxiv_id}: {content_type}")
        os.makedirs(download_dir, exist_ok=True)
        tar_path = os.path.join(download_dir, f"{arxiv_id}.tar.gz")
        with open(tar_path, 'wb') as f:
            f.write(response.content)
        return tar_path
    else:
        print(f"Failed to download {arxiv_id}")
        return None


def extract_image_paths_from_figure_env(tex_file, extract_dir):
    """Extract image paths from LaTeX figure environments in the source file."""
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    image_paths = []
    encodings = ['utf-8', 'latin1', 'ascii']

    try:
        with tarfile.open(tex_file) as tar:
            tar.extractall(path=extract_dir)
    except tarfile.ReadError:
        print("The downloaded file is not a valid tar.gz archive.")
        return image_paths

    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith(".tex"):
                tex_file_path = os.path.join(root, file)
                for encoding in encodings:
                    try:
                        with open(tex_file_path, 'r', encoding=encoding) as f:
                            tex_content = f.read()
                            figures = re.findall(r'\\begin{figure}.*?\\end{figure}', tex_content, re.DOTALL)
                            for figure in figures:
                                image_paths += re.findall(r'\\includegraphics\[.*?\]{(.*?)}', figure)
                        break
                    except UnicodeDecodeError:
                        print(f"Error decoding {tex_file_path} with {encoding} encoding. Trying next encoding.")
                    except Exception as e:
                        print(f"An unexpected error occurred while processing {tex_file_path}: {e}")
                        break

    return image_paths


def download_random_top_image(image_paths, article_id, extract_dir, download_dir):
    """Download a random top image from the list of image paths."""
    article_download_dir = os.path.join(download_dir, article_id)
    if not os.path.exists(article_download_dir):
        os.makedirs(article_download_dir)

    if not image_paths:
        print(f"No images found in the article {article_id}.")
        # No images found, try extracting from PDF
        pdf_path = download_pdf(article_id, extract_dir)
        if pdf_path:
            extracted_images = extract_images_from_pdf(pdf_path, extract_dir)
            if extracted_images:
                selected_image = random.choice(extracted_images)
                shutil.copy(selected_image, article_download_dir)
                print(f"Saved {os.path.basename(selected_image)} from PDF extraction.")
        return

    # Count the frequency of each image
    image_counter = Counter(image_paths)
    # Get the top 5 most referenced images that are PDF or PNG
    top_images = [img for img, count in image_counter.most_common() if img.endswith('.pdf') or img.endswith('.png')][:5]

    if not top_images:
        print(f"No PDF or PNG images found in the top references for article {article_id}.")
        # No PDF or PNG images found, try extracting from PDF
        pdf_path = download_pdf(article_id, extract_dir)
        if pdf_path:
            extracted_images = extract_images_from_pdf(pdf_path, extract_dir)
            if extracted_images:
                selected_image = random.choice(extracted_images)
                shutil.copy(selected_image, article_download_dir)
                print(f"Saved {os.path.basename(selected_image)} from PDF extraction.")
        return

    # Randomly select one image from the top 5
    selected_image = random.choice(top_images)
    # Check if the selected image file exists
    full_img_path = os.path.abspath(os.path.join(extract_dir, selected_image))
    if os.path.exists(full_img_path):
        img_name = os.path.basename(selected_image)
        dest_img_path = os.path.join(article_download_dir, img_name)
        with open(full_img_path, 'rb') as src, open(dest_img_path, 'wb') as dest:
            dest.write(src.read())
        print(f"Saved {img_name} of {len(image_paths)}.")
        # Clean up: remove all other files in the article's directory
        for item in os.listdir(article_download_dir):
            item_path = os.path.join(article_download_dir, item)
            if item_path != dest_img_path:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
    else:
        print(f"Selected image not found: {full_img_path}")


def download_pdf(arxiv_id, output_dir):
    """Download the PDF file of an arxiv article by ID."""
    url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to download PDF for {arxiv_id}")
        return None

    pdf_path = os.path.join(output_dir, f'{arxiv_id}.pdf')
    with open(pdf_path, 'wb') as file:
        file.write(response.content)
    return pdf_path


def image_hash(image_path):
    """Generate a hash for an image file."""
    with open(image_path, 'rb') as f:
        img_hash = hashlib.md5(f.read()).hexdigest()
    return img_hash


def contains_text_or_graph(img_path):
    """Check if an image contains text or graphs."""
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return False

    return True


def extract_images_from_pdf(pdf_path, output_dir, min_image_area=10000):
    """Extract images from a PDF file."""
    pdf_document = fitz.open(pdf_path)
    image_paths = []
    unique_hashes = set()

    for page_num in range(len(pdf_document)):
        try:
            page = pdf_document.load_page(page_num)
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                img_path = os.path.join(output_dir,
                                        f'{os.path.basename(pdf_path).split(".")[0]}_page{page_num + 1}_img{img_index + 1}.{image_ext}')

                with open(img_path, 'wb') as img_file:
                    img_file.write(image_bytes)

                # Check for duplicate images
                img_hash = image_hash(img_path)
                if img_hash in unique_hashes:
                    os.remove(img_path)
                    continue
                unique_hashes.add(img_hash)

                # Check if the image has information (filter by area)
                with Image.open(img_path) as img:
                    if img.size[0] * img.size[1] < min_image_area:
                        os.remove(img_path)
                        continue

                # Check if the image contains text or graphs
                if not contains_text_or_graph(img_path):
                    os.remove(img_path)
                    continue

                image_paths.append(img_path)
        except Exception as e:
            print(f"Error processing page {page_num} of {pdf_path}: {e}")
            continue

    return image_paths


def normalize_text(s):
    """Normalize text by removing numbers, punctuation, and converting to lowercase."""
    s = ''.join([i for i in s if not i.isdigit()])  # Remove numbers
    s = s.replace('-', ' ')
    s = s.replace('/', ' ')
    s = s.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    s = s.lower()  # Convert to lowercase
    s = s.replace('  ', ' ')  # Remove double spaces
    return s


# Load machine learning models
model = joblib.load('/Users/jiaohuangbixia/Downloads/1/MODELTRAIN/relevance_model4.pkl')

# Initiate vectorizer
vectorizer = HashingVectorizer(ngram_range=(1, 3))


def calculate_relevance(title, abstract):
    """Calculate the relevance of an article using a pre-trained model."""
    text = normalize_text(title + ' ' + abstract)
    features = vectorizer.transform([text])
    relevance = model.predict(features)[0]

    # Dark matter filtration
    if re.search(r'\bdark matter\b', text, re.IGNORECASE):
        relevance = 90 + 0.1 * relevance

    # 100/0
    if relevance > 100:
        relevance = 100
    elif relevance < 0:
        relevance = 0

    return relevance


def extract_authors_from_tex(tex_file, extract_dir):
    """Extract authors from LaTeX source file."""
    authors = []
    encodings = ['utf-8', 'latin1', 'ascii']

    try:
        with tarfile.open(tex_file) as tar:
            tar.extractall(path=extract_dir)
    except tarfile.ReadError:
        print("The downloaded file is not a valid tar.gz archive.")
        return None

    def find_matching_brace(s, start):
        """Find the index of the matching closing brace for the brace at index start."""
        stack = []
        for i in range(start, len(s)):
            if s[i] == '{':
                stack.append('{')
            elif s[i] == '}':
                stack.pop()
                if not stack:
                    return i
        return -1  # No matching closing brace found

    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith(".tex"):
                tex_file_path = os.path.join(root, file)
                for encoding in encodings:
                    try:
                        with open(tex_file_path, 'r', encoding=encoding) as f:
                            tex_content = f.read()

                            # Combine lines to handle multi-line \author definitions
                            tex_content = tex_content.replace('\n', ' ')

                            # Find all occurrences of \author{...}
                            author_sections = []
                            pattern = r'\\author(?:\[[^\]]*\])?{'
                            for match in re.finditer(pattern, tex_content):
                                start_index = match.end() - 1
                                end_index = find_matching_brace(tex_content, start_index)
                                if end_index != -1:
                                    authors_content = tex_content[start_index+1:end_index]
                                    author_sections.append(authors_content)

                            if not author_sections:
                                continue  # No \author sections found in this file

                            # Process each \author{...} section
                            for authors_content in author_sections:
                                # Remove LaTeX commands and comments
                                authors_content = re.sub(r'%.*', '', authors_content)
                                authors_content = re.sub(r'\\thanks{.*?}', '', authors_content, flags=re.DOTALL)
                                authors_content = re.sub(r'\\email{.*?}', '', authors_content, flags=re.DOTALL)
                                authors_content = re.sub(r'\\orcidlink{.*?}', '', authors_content, flags=re.DOTALL)
                                authors_content = re.sub(r'\\(newauthor|and)', ',', authors_content)
                                authors_content = re.sub(r'\\[a-zA-Z]+', '', authors_content)
                                authors_content = re.sub(r'\$.*?\$', '', authors_content, flags=re.DOTALL)  # Remove inline math
                                authors_content = re.sub(r'\{.*?\}', '', authors_content, flags=re.DOTALL)  # Remove nested braces
                                authors_content = re.sub(r'\s+', ' ', authors_content)  # Normalize whitespace
                                authors_content = authors_content.strip(', ')

                                # Split authors by commas
                                authors_list = [author.strip() for author in authors_content.split(',') if author.strip()]
                                authors.extend(authors_list)

                            if authors:
                                # Remove duplicates while preserving order
                                seen = set()
                                unique_authors = []
                                for author in authors:
                                    # Remove email addresses and URLs
                                    author = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '', author)
                                    author = re.sub(r'http\S+', '', author)
                                    author = author.strip()
                                    if author and author not in seen:
                                        seen.add(author)
                                        unique_authors.append(author)
                                return unique_authors
                    except UnicodeDecodeError:
                        print(f"Error decoding {tex_file_path} with {encoding} encoding. Trying next encoding.")
                    except Exception as e:
                        print(f"An unexpected error occurred while processing {tex_file_path}: {e}")
                        break

    return None


# Added function to scrape authors from arXiv page
def scrape_authors_from_arxiv_page(arxiv_id):
    """Scrape authors from the arXiv abstract page."""
    url = f'https://arxiv.org/abs/{arxiv_id}'
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch page for arxiv_id {arxiv_id}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    authors_div = soup.find('div', class_='authors')
    if not authors_div:
        print(f"Authors section not found for arxiv_id {arxiv_id}")
        return None

    authors = authors_div.find_all('a')
    authors_list = [author.get_text(strip=True) for author in authors]
    return authors_list if authors_list else None


if __name__ == '__main__':
    categories = ['astro-ph', 'gr-qc', 'hep-ex', 'hep-ph', 'hep-th']
    start = datetime.now()  # Record Start Time

    all_articles = []  # Save all articles
    filtered_articles = []  # Save filtered articles

    processed_ids = load_processed_ids()  # Load processed ids

    for category in categories:  # Iterate through each category
        arxiv_ids = fetch_arxiv_new_articles(category)  # Get new article IDs for the category
        new_arxiv_ids = [arxiv_id for arxiv_id in arxiv_ids if arxiv_id not in processed_ids]  # Filter out processed IDs
        for arxiv_id in new_arxiv_ids:  # Process each new article ID
            print(f"Processing arxiv_id: {arxiv_id}")
            details = fetch_article_details(arxiv_id)  # Fetch article details
            if not details:  # If no details, skip
                continue

            relevance = calculate_relevance(details['title'], details['abstract'])  # Calculate relevance score
            print(f"Predicted relevance for arxiv_id {arxiv_id}: {relevance}")

            if relevance > 50:  # If relevance score is greater than 50
                article = {
                    'relevance': relevance,  # Add relevance score
                    'category': category,
                    'title': details['title'],
                    'authors': 'N/A',  # Initially set to 'N/A'
                    'summary': details['abstract'],
                    'arxiv_id': details['arxiv_id'],
                    'pdf_url': details['pdf_url'],
                    'html_url': details['html_url'],
                    'source_url': details['pdf_url'],
                    'image_paths': '[]'
                }
                filtered_articles.append(article)

        print(f'{len(new_arxiv_ids)} new titles saved from Arxiv {category}.')
        # For filtered articles, individually query author information via arXiv API
        for article in filtered_articles:
            arxiv_id = article['arxiv_id']
            attempts = 0
            while article['authors'] == 'N/A' and attempts < 3:  # Maximum 3 attempts
                attempts += 1
                print(f"Attempting to fetch authors for arxiv_id: {arxiv_id}, attempt {attempts}")
                authors = fetch_author_info_arxiv_api(arxiv_id)
                if authors:
                    article['authors'] = ', '.join(authors)
                else:
                    print(f"Attempt {attempts} failed for arxiv_id: {arxiv_id}")

            if article['authors'] == 'N/A':
                print(f"API failed, attempting to extract authors from LaTeX source for arxiv_id: {arxiv_id}")
                temp_dir = os.path.join('./temp_tex', arxiv_id)
                create_directory(temp_dir)
                tex_tar_path = download_tex_from_arxiv(arxiv_id, temp_dir)

                if tex_tar_path:
                    authors_from_tex = extract_authors_from_tex(tex_tar_path, temp_dir)
                    if authors_from_tex:
                        article['authors'] = ', '.join(authors_from_tex)
                        print(f"Successfully extracted authors from LaTeX: {article['authors']}")
                    else:
                        print(f"Failed to extract authors from LaTeX for arxiv_id: {arxiv_id}")

                # Clean up temporary directory
                shutil.rmtree(temp_dir)

            # Additional attempt to scrape authors from arXiv page
            if article['authors'] == 'N/A':
                print(f"Failed to retrieve authors via API and LaTeX. Attempting to scrape authors from the arXiv page for arxiv_id: {arxiv_id}")
                authors_from_page = scrape_authors_from_arxiv_page(arxiv_id)
                if authors_from_page:
                    article['authors'] = ', '.join(authors_from_page)
                    print(f"Successfully scraped authors from arXiv page: {article['authors']}")
                else:
                    print(f"Failed to scrape authors from arXiv page for arxiv_id: {arxiv_id}")

        end = datetime.now()  # Record end time
        print(f'Time elapsed: {timedelta(seconds=round((end - start).total_seconds()))}')

        # Fetch images and update image paths
        if filtered_articles:
            for article in filtered_articles:
                arxiv_id = article['arxiv_id']
                print(f"Fetching images for article: {arxiv_id}")
                article_output_dir = os.path.join('./viewable_images', arxiv_id)
                create_directory(article_output_dir)

                # Download tar file to a temporary directory
                temp_dir = os.path.join(article_output_dir, 'temp')
                create_directory(temp_dir)
                tex_tar_path = download_tex_from_arxiv(arxiv_id, temp_dir)

                if tex_tar_path:
                    tex_images = extract_image_paths_from_figure_env(tex_tar_path, temp_dir)
                    if not tex_images:  # If no images found in .tex files, try extracting from PDF
                        pdf_path = download_pdf(arxiv_id, temp_dir)
                        if pdf_path:
                            extracted_images = extract_images_from_pdf(pdf_path, article_output_dir)
                            if extracted_images:
                                article['image_paths'] = extracted_images[0]  # Save only one image
                            else:  # If no images extracted from PDF, randomly pick one image from the output directory
                                all_images = [f for f in os.listdir(article_output_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
                                if all_images:
                                    article['image_paths'] = os.path.join(article_output_dir, random.choice(all_images))
                    else:
                        download_random_top_image(tex_images, arxiv_id, temp_dir, article_output_dir)
                        article['image_paths'] = os.listdir(article_output_dir)  # Update image_paths

                # Clean up the temporary directory
                shutil.rmtree(temp_dir)

        # Save filtered articles to a new CSV file
        df = pd.DataFrame(filtered_articles)
        df.to_csv('filtered.csv', index=False, encoding='utf-8')

        # Print summary of processed and saved articles
        total_processed_articles = sum([len(fetch_arxiv_new_articles(category)) for category in categories])
        total_saved_articles = len(filtered_articles)
        print(f"Processed {total_processed_articles} articles, saved {total_saved_articles} articles to 'filtered.csv'.")

        # Update the list of processed article IDs
        processed_ids.update(new_arxiv_ids)
        save_processed_ids(processed_ids)

    print("Finished Scrape_Daily_New.py")
