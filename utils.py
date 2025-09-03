from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import requests
from bs4 import BeautifulSoup

FB_RENDER_WAIT = 5
REQUEST_TIMEOUT = 10
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())

def get_web_content(url: str, timeout: int = REQUEST_TIMEOUT) -> str:
    try:
        headers = {"User-Agent": USER_AGENT}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return clean_html(r.text)
    except Exception as e:
        return f"[Lỗi khi truy cập {url}: {e}]"

def get_facebook_content(url: str, headless: bool = True, wait: int = FB_RENDER_WAIT) -> str:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"user-agent={USER_AGENT}")
    driver = None
    try:
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(30)
        driver.get(url)
        time.sleep(wait)
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        texts = []
        for el in soup.find_all(["p", "div", "span"]):
            t = el.get_text(separator=" ", strip=True)
            if t:
                texts.append(t)
        return " ".join(texts)
    except Exception as e:
        return f"[Lỗi khi truy xuất Facebook {url}: {e}]"
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass

def fetch_page_content(url: str) -> str:
    u = (url or "").lower()
    if "facebook.com" in u:
        return get_facebook_content(url, headless=True)
    return get_web_content(url)