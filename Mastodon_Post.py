import os
import re
import time
import random
import pandas as pd
from mastodon import Mastodon
from PIL import Image
import signal
import imagehash
from pdf2image import convert_from_path

# Use your original working directory
os.chdir('/Users/jiaohuangbixia/Downloads/1/MODELTRAIN')
print("Starting Mastodon_Post.py")

# Use your actual Mastodon instance and access token
mastodon_instance = 'https://mastodon.social'
access_token = '_Jx-2_Lu24mM4RfMPkV121yGVBfAR39x1Cc8puSTUto'  # Replace with your actual access token

mastodon = Mastodon(access_token=access_token, api_base_url=mastodon_instance, request_timeout=30)
published_titles_file = 'published_arxiv_titles.csv'

def load_published_titles():
    """Load the list of published titles from a CSV file."""
    if os.path.exists(published_titles_file):
        return set(pd.read_csv(published_titles_file)['title'].tolist())
    else:
        return set()

def save_published_title(title):
    """Save a new published title to the CSV file."""
    new_entry = pd.DataFrame([[title]], columns=['title'])
    if not os.path.exists(published_titles_file):
        new_entry.to_csv(published_titles_file, index=False)
    else:
        df = pd.read_csv(published_titles_file)
        if title not in df['title'].values:
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_csv(published_titles_file, index=False)

def compress_image(image_path, output_path, quality=70):
    """Compress an image to reduce file size."""
    try:
        image = Image.open(image_path)
        if image.mode == 'RGBA':
            image = image.convert('RGB')  # Convert to RGB mode
        image.save(output_path, "JPEG", quality=quality)
        return output_path
    except Exception as e:
        print(f"Error compressing image: {e}")
        return None

def clean_text(text):
    """Clean up the text by removing extra spaces and special characters."""
    if isinstance(text, float):
        text = ""
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('$', '').replace('_', '')
    return text.strip()

def timeout_handler(signum, frame):
    raise TimeoutError

def post_update_with_images(message, image_paths):
    """Post an update with images to Mastodon."""
    media_ids = []
    upload_delay = 10

    for image_path in image_paths:  # Only post one image
        if os.path.exists(image_path):
            compressed_image_path = compress_image(image_path, os.path.splitext(image_path)[0] + '_compressed.jpg')
            if compressed_image_path:
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(120)
                    print(f"Uploading image: {compressed_image_path}")
                    media = mastodon.media_post(compressed_image_path)
                    media_ids.append(media['id'])
                    signal.alarm(0)
                    print(f"Uploaded image: {compressed_image_path}")
                except TimeoutError:
                    print(f"Uploading image timed out: {compressed_image_path}")
                    signal.alarm(0)
                except Exception as e:
                    print(f"Error uploading media: {e}")
                    signal.alarm(0)

                time.sleep(upload_delay)
            else:
                print(f"Image compression failed: {image_path}")
        else:
            print(f"Image not found: {image_path}")

        break  # Ensure only one image is posted

    try:
        if len(message) > 500:
            message = message[:497] + '...'

        if media_ids:
            print(f"Posting toot with message: {message} and images")
            mastodon.status_post(status=message, media_ids=media_ids)
            print(f"Posted toot with message: {message} and images: {image_paths}")
        else:
            print(f"No images to post for message: {message}. Skipping post.")
    except Exception as e:
        print(f"Error posting to Mastodon: {e}")

def convert_pdf_to_images(pdf_path, output_dir):
    """Convert a PDF file to images."""
    poppler_path = '/opt/homebrew/bin'  # Use your original poppler path
    images = convert_from_path(pdf_path, poppler_path=poppler_path)
    image_paths = []
    for i, image in enumerate(images):
        output_path = os.path.join(output_dir, f'page_{i+1}.jpg')
        image.save(output_path, 'JPEG')
        image_paths.append(output_path)
    return image_paths

def process_authors(authors):
    """Process the authors list to retain up to three names and append 'et al' if necessary."""
    # Remove LaTeX commands and any text after '\\' (assuming institutions come after '\\')
    authors_cleaned = re.sub(r'\\[^\s]*', '', authors)
    authors_cleaned = authors_cleaned.split('\\')[0]
    # Remove any remaining LaTeX braces
    authors_cleaned = authors_cleaned.replace('{', '').replace('}', '')
    # Normalize whitespace
    authors_cleaned = re.sub(r'\s+', ' ', authors_cleaned)
    # Split authors by comma
    authors_list = [author.strip() for author in authors_cleaned.split(',') if author.strip()]
    # Further split authors that are connected with 'and' within the list
    expanded_authors_list = []
    for author in authors_list:
        if ' and ' in author:
            expanded_authors_list.extend([a.strip() for a in author.split(' and ')])
        else:
            expanded_authors_list.append(author)
    # Limit to three authors and add 'et al' if necessary
    if len(expanded_authors_list) > 3:
        authors = ', '.join(expanded_authors_list[:3]) + ' et al'
    else:
        authors = ', '.join(expanded_authors_list)
    return authors

def main():
    viewable_images_dir = '/Users/jiaohuangbixia/Downloads/1/MODELTRAIN/viewable_images'

    published_titles = load_published_titles()
    print(f"Loaded published titles: {published_titles}")

    # Read CSV file
    csv_file_path = '/Users/jiaohuangbixia/Downloads/1/MODELTRAIN/filtered.csv'
    df = pd.read_csv(csv_file_path, dtype={'arxiv_id': str})

    for index, row in df.iterrows():
        title = clean_text(row['title'])
        relevance = float(row['relevance'])
        if relevance < 0:
            relevance = 0
        elif relevance > 100:
            relevance = 100
        relevance = f"{relevance:.2f}"  # Format to two decimal places

        authors = clean_text(row['authors'])
        pdf_url = row['pdf_url'].strip()  # Read PDF URL from CSV
        summary = clean_text(row['summary'])
        arxiv_id = row['arxiv_id'].strip()

        print(f"Processing article: {title}")

        if title in published_titles:
            print(f"Article titled '{title}' has already been published, skipping.")
            continue

        # Check if authors is 'N/A' and skip publishing if so
        if authors == 'N/A':
            print(f"Article titled '{title}' has authors 'N/A', skipping.")
            continue

        # Process authors
        authors = process_authors(authors)

        # Find images in the specified directory for the article
        image_dir = os.path.join(viewable_images_dir, arxiv_id)
        image_paths = []
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(root, file))
                elif file.endswith('.pdf'):
                    pdf_images = convert_pdf_to_images(os.path.join(root, file), root)
                    image_paths.extend(pdf_images)

        if not image_paths:
            print(f"No images found for article titled '{title}', skipping post.")
            continue

        # If more than one image, randomly select one
        if len(image_paths) > 1:
            image_paths = [random.choice(image_paths)]

        valid_image_paths = []
        seen_hashes = set()
        for img_path in image_paths:
            if os.path.exists(img_path):
                img_hash = imagehash.average_hash(Image.open(img_path))
                if img_hash not in seen_hashes:
                    seen_hashes.add(img_hash)
                    valid_image_paths.append(img_path)
                else:
                    print(f"Duplicate image detected: {img_path}")
            else:
                print(f"Image path does not exist: {img_path}")

        message = f"Relevance: {relevance}%\nTitle: {title}\nPDF URL: {pdf_url}\nAuthors: {authors}\nSummary: {summary}"

        print(f"Posting article: {title}")
        print(f"Message: {message}")
        print(f"Image paths: {valid_image_paths}")
        post_update_with_images(message, valid_image_paths)
        print(f"Finished posting article: {title}")

        save_published_title(title)
        published_titles.add(title)
        print(f"Saved published title: {title}")

        time.sleep(10)

if __name__ == "__main__":
    main()
print("Finished Mastodon_Post.py")
