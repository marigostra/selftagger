import requests
import  json
from urllib.parse import quote

from selftagger.conf import Config

class SearX:
    def __init__(self):
        self.conf = Config()
    
    def search(query):
        q = quote(query)
        response = requests.get(f"{conf.searxUrl}/search?q={q}&format=json")
        response.raise_for_status()
        data = json.loads(response.text)
        data = data['results']
        print(f"Найдено {len(data)} страниц")
        res = ""
        for i in data:
            url = i["url"]
            title = i["title"]
            text = i["content"]
            res = res + f"Страница {url} с заголовком {title} содержит текст: {text}. "
            return res

    def search_paper(query):
        q = quote(f"!science {query}")
        response = requests.get(f"{conf.searxUrl}/search?q={q}&format=json")
        response.raise_for_status()
        data = json.loads(response.text)
        data = data['results']
        res = ""
        for i in data:
            pdfUrl = i["pdf_url"]  if "pdf_url" in i else ""
            htmlUrl = i["html_url"]  if "html_url" in i else ""
            if not pdfUrl:
                url = htmlUrl
            else:
                url = pdfUrl
            authors = ""
            if "authors" in i:
                authors = " авторы "
                for a in i["authors"]:
                    authors = authors + f"{a},"
            date = i["publishedDate"]
            title = i["title"]
            text = i["content"]
            info = f"Статья с названием \"{title}\", {authors} опубликованная {date}, содержит текст\": {text}\" и доступна по ссылке {url}."
            res = res + info
        return res

