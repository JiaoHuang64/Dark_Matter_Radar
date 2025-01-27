import os
import re
import time
import random
import pandas as pd
from PIL import Image
import requests
from pdf2image import convert_from_path

# Use your original working directory
os.chdir('/Users/jiaohuangbixia/Downloads/1/MODELTRAIN')
print("Starting WordPress_Post.py")

# WordPress REST API details
wordpress_api_url = 'https://darkmatterradar.com/wp-json/wp/v2/posts'
wordpress_username = 'jiaohuang780@gmail.com'
wordpress_password = 'R99EqMfad91JoD3mysRhq3pt'  # Replace with your actual password

# Published titles tracking file
published_titles_file = 'published_wp_titles.csv'

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
    authors_cleaned = re.sub(r'\\[^\s]*', '', authors)
    authors_cleaned = authors_cleaned.split('\\')[0]
    authors_cleaned = authors_cleaned.replace('{', '').replace('}', '')
    authors_cleaned = re.sub(r'\s+', ' ', authors_cleaned)
    authors_list = [author.strip() for author in authors_cleaned.split(',') if author.strip()]
    expanded_authors_list = []
    for author in authors_list:
        if ' and ' in author:
            expanded_authors_list.extend([a.strip() for a in author.split(' and ')])
        else:
            expanded_authors_list.append(author)
    if len(expanded_authors_list) > 3:
        authors = ', '.join(expanded_authors_list[:3]) + ' et al'
    else:
        authors = ', '.join(expanded_authors_list)
    return authors

def post_to_wordpress(title, content, image_path=None):
    """Post an article to WordPress."""
    try:
        # Authenticate and set up headers
        credentials = (wordpress_username, wordpress_password)
        headers = {
            'Content-Type': 'application/json'
        }

        # Upload image if available
        media_id = None
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as img:
                media_upload_response = requests.post(
                    f"{wordpress_api_url.replace('/posts', '/media')}",
                    auth=credentials,
                    files={'file': img}
                )
                if media_upload_response.status_code == 201:
                    media_id = media_upload_response.json().get('id')
                    print(f"Image uploaded successfully with media ID: {media_id}")
                else:
                    print(f"Failed to upload image: {media_upload_response.text}")

        # Prepare post payload
        post_data = {
            'title': title,
            'content': content,
            'status': 'publish'
        }
        if media_id:
            post_data['featured_media'] = media_id

        # Post to WordPress
        response = requests.post(wordpress_api_url, auth=credentials, json=post_data)
        if response.status_code == 201:
            print(f"Post published successfully: {response.json().get('link')}")
        else:
            print(f"Failed to publish post: {response.text}")

    except Exception as e:
        print(f"Error posting to WordPress: {e}")

def main():
    viewable_images_dir = '/Users/jiaohuangbixia/Downloads/1/MODELTRAIN/viewable_images'

    # Load published titles to avoid duplicates
    published_titles = load_published_titles()
    print(f"Loaded published titles: {published_titles}")

    # Read CSV file
    csv_file_path = '/Users/jiaohuangbixia/Downloads/1/MODELTRAIN/filtered.csv'
    df = pd.read_csv(csv_file_path, dtype={'arxiv_id': str})

    for index, row in df.iterrows():
        title = clean_text(row['title'])

        # Skip already published titles
        if title in published_titles:
            print(f"Article titled '{title}' has already been published, skipping.")
            continue

        relevance = float(row['relevance'])
        if relevance < 0:
            relevance = 0
        elif relevance > 100:
            relevance = 100
        relevance = f"{relevance:.2f}"

        authors = clean_text(row['authors'])
        pdf_url = row['pdf_url'].strip()
        summary = clean_text(row['summary'])
        arxiv_id = row['arxiv_id'].strip()

        print(f"Processing article: {title}")

        if authors == 'N/A':
            print(f"Article titled '{title}' has authors 'N/A', skipping.")
            continue

        authors = process_authors(authors)

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

        if len(image_paths) > 1:
            image_paths = [random.choice(image_paths)]

        message = f"""
        <strong>Relevance:</strong> {relevance}%<br>
        <strong>Title:</strong> {title}<br>
        <strong>PDF URL:</strong> <a href="{pdf_url}">{pdf_url}</a><br>
        <strong>Authors:</strong> {authors}<br>
        <strong>Summary:</strong> {summary}
        """

        print(f"Posting article: {title}")
        print(f"Message: {message}")
        print(f"Image path: {image_paths[0] if image_paths else None}")
        post_to_wordpress(title, message, image_paths[0] if image_paths else None)
        print(f"Finished posting article: {title}")

        # Save the published title
        save_published_title(title)
        published_titles.add(title)
        print(f"Saved published title: {title}")

        time.sleep(10)

if __name__ == "__main__":
    main()
print("Finished WordPress_Post.py")
