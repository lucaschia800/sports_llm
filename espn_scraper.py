import aiohttp
from bs4 import BeautifulSoup
import asyncio
import pandas as pd
import json


async def fetch(client_session, url):
  headers = {
          "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
          "Referer": "https://www.espn.com/",
      }
  async with client_session.get(url, headers=headers, ssl=False) as response:  # 👈 Disables SSL verification
    return await response.text()

async def parse(html_content):
  if not html_content:
      return "recap not found"
  soup = BeautifulSoup(html_content, "html.parser")
  # Extract all paragraph text
  paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
  # Join them into a single text block
  cleaned_text = "\n\n".join(paragraphs)
  return cleaned_text

async def fetch_parse_write(session, game_id, sport):
  url = f"https://www.espn.com/{sport}/recap?gameId={game_id}"
  """Fetch, parse, and return structured data."""
  html_text = await fetch(session, url)
  content = await parse(html_text)
  return {"url": url, "content": content}

async def main(game_ids, filename, sport):
  client_session = aiohttp.ClientSession()
  """Scrape multiple articles asynchronously and save to CSV."""
  async with aiohttp.ClientSession() as session:
      tasks = [fetch_parse_write(session, id, sport) for id in game_ids]
      results = await asyncio.gather(*tasks)

      # Convert results to DataFrame
      df = pd.DataFrame(results)
      df.to_csv(filename, index=False)
      return df

# Example usage
#2024 -> 670
#2023 ->547

sports = ['nfl', 'nba', 'nhl']

async def process_sports():
  for sport in sports:
    game_ids = json.load(open(f"{sport}_game_ids.json"))  # Load the correct game IDs file
    filename = f"{sport}_espn_articles.csv"
    await main(game_ids, filename, sport)  # Ensure async function is awaited

if __name__ == "__main__":
  asyncio.run(process_sports())  # Run the loop properly