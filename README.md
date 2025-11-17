# Amazon Product Scraper
A simple Amazon product scraping tool that extracts product details such as Title, Price, Rating, Reviews Count, and Product Link. Works in Google Colab and saves results to CSV format.

## Features
- Scrapes multiple pages
- Takes search term input
- Outputs CSV file
- Works in Colab / Jupyter / Local Python

## How to Run (Google Colab)

1. Upload the notebook `amazon_scraper.ipynb` to Google Colab  
2. Run all cells  
3. Enter the product name when prompted  
4. CSV file will auto-download



# Face Authentication System 
A face verification system that accepts two images, extracts facial embeddings, and returns whether they belong to the same person along with a similarity score.

## Features
- Face detection
- Embedding generation using InsightFace
- Similarity score between two images
- Returns:
  - Same Person / Different Person
  - Similarity %
  - Bounding Boxes


## Example Usage

This example demonstrates how to use the Face Authentication API to verify if two face images belong to the same person.

### Request

**POST** `http://127.0.0.1:8000/authenticate`

**Request Body (JSON):**
```json
{
  "image_url_1": "https://image.tmdb.org/t/p/w400/nzz4NZfJwX9eGxFLNXaNn5YCk34.jpg",
  "image_url_2": "https://images.news18.com/ibnlive/uploads/2021/11/saurabh-raj-jain-16382814474x3.jpg",
  "threshold": 0.35
}



