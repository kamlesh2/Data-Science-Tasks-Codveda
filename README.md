📘 Data Science Task Report
Author: Kamlesh 
Duration: 24-09-2025 to 24-10-2025 
Tools Used: Python, pandas, scikit-learn, statsmodels, TensorFlow, Keras, matplotlib, seaborn, BeautifulSoup, requests 
Data Sources: Wikipedia, local CSVs, Yahoo Finance, Fashion MNIST .ubyte files

🔹 Level 1: Basic
✅ Task 1: Web Scraping
Scraped Oscar-winning film data from Wikipedia.

Code:# Web scraping Oscar winners
import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://en.wikipedia.org/wiki/List_of_Academy_Award-winning_films"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

table = soup.find("table", {"class": "wikitable"})
rows = table.find_all("tr")[1:]

data = []
for row in rows:
    cols = row.find_all(["th", "td"])
    title = cols[0].text.strip()
    awards = int(cols[1].text.strip())
    nominations = int(cols[2].text.strip())
    year = cols[3].text.strip()
    data.append([title, awards, nominations, year])

df = pd.DataFrame(data, columns=["Title", "Awards", "Nominations", "Year"])
df.to_csv("oscar_wikipedia_films.csv", index=False)

Result: 
✔ Scraped 150+ films from 1964–2024 
✔ Saved to oscar_wikipedia_films.csv
