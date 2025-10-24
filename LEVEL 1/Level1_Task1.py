from playwright.sync_api import sync_playwright
import csv

def scrape_wikipedia_oscar_films():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("https://en.wikipedia.org/wiki/List_of_Academy_Award-winning_films")
        page.wait_for_selector("table.wikitable", timeout=10000)
        rows = page.query_selector_all("table.wikitable tbody tr")[1:]
        data = []
        for row in rows:
            cells = row.query_selector_all("td")
            if len(cells) >= 3:
                title = cells[0].inner_text().strip()
                awards = cells[1].inner_text().strip()
                nominations = cells[2].inner_text().strip()
                print(f"{title} - Awards: {awards}, Nominations: {nominations}")
                data.append([title, awards, nominations])
        with open("oscar_wikipedia_films.csv", mode="w", newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Title", "Awards", "Nominations"])
            writer.writerows(data)
        print("Data saved to oscar_wikipedia_films.csv")
        browser.close()

scrape_wikipedia_oscar_films()
