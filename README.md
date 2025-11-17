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


## Project Structure
```
Amazon-Scraper/
│── amazon_scraper.ipynb
│── requirements_scrapper.txt
│── README.md
│── sample.csv   
```




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
 
  ## Run Server
  uvicorn main:app --reload

  ## Project Structure
  ```
  Face-Authentication/
│── main.py
│── models/
│── embeddings/
│── requirements.txt
│── README.md
```

