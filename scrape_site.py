import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def save_website_content():
    base_url = "https://www.sofabed.com/collections/luonto-king-sleeper-sofa/products/belton-king-sofa-sleeper-with-manual-or-power-option"
    visited_urls = set()
    all_content = []

    def scrape_page(url):
        if url in visited_urls:
            return
        
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text content from main content areas
            # Modify these selectors based on your website's structure
            content_areas = soup.find_all(['p', 'h1', 'h2', 'h3', 'div.product-description'])
            
            for content in content_areas:
                text = content.get_text(strip=True)
                if text:
                    all_content.append(text)
            
            visited_urls.add(url)

        except Exception as e:
            print(f"Error scraping {url}: {e}")

    # Start with the main page
    scrape_page(base_url)

    # Save all content to file
    with open('website_content2.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_content))

if __name__ == "__main__":
    save_website_content() 