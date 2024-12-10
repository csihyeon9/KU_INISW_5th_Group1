# src/fetch_content.py
import requests
from bs4 import BeautifulSoup

def fetch_content(url):
    try:
        # 웹페이지 요청
        response = requests.get(url)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생

        # HTML 파싱
        soup = BeautifulSoup(response.text, 'html.parser')

        if "hankyung.com" in url:
            return fetch_hankyung_content(soup, url)
        elif "mk.co.kr" in url:
            return fetch_maeil_content(soup, url)
        elif "naver.com" in url:
            return fetch_naver_content(soup, url)
        elif "daum.net" in url:
            return fetch_daum_content(soup, url)
        else:
            return "Unsupported URL"

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch the URL: {e}")


def fetch_hankyung_content(soup, url):
    """
    Fetches the content from 한국경제 articles.
    """
    # 본문 내용 가져오기
    content_element = soup.find('div', class_='article-body', id='articletxt', itemprop='articleBody')
    content = content_element.get_text(strip=True) if content_element else "No content found on Hankyung."

    # 제목 가져오기
    title_element = soup.find('h1', class_='headline')
    title = title_element.get_text(strip=True) if title_element else "No title found."

    # 날짜 가져오기
    date_element = soup.find('span', class_='txt-date')
    date = date_element.get_text(strip=True) if date_element else "No date found."

    return {
        "content": content,
        "title": title,
        "date": date,
        "url": url
    }


def fetch_maeil_content(soup, url):
    """
    Fetches the content from 매일경제 articles (e.g., economy section).
    """
    # 본문 내용 가져오기
    content_element = soup.find('div', class_='news_cnt_detail_wrap')
    content = content_element.get_text(strip=True) if content_element else "No content found on Maeil."

    # 제목 가져오기
    title_element = soup.find('h2', class_='news_ttl')
    title = title_element.get_text(strip=True) if title_element else "No title found."

    # 날짜 가져오기
    date_element = soup.find('dl', class_='registration').find('dd')
    date = date_element.get_text(strip=True) if date_element else "No date found."

    return {
        "content": content,
        "title": title,
        "date": date,
        "url": url
    }


def fetch_naver_content(soup, url):
    """
    Fetches the content from 네이버 articles.
    """
    # 본문 내용 가져오기
    content_element = soup.find('article', id='dic_area', class_='go_trans _article_content')
    content = content_element.get_text(strip=True) if content_element else "No content found on Naver."

    # 제목 가져오기
    title_element = soup.find('h2', id='title_area', class_='media_end_head_headline')
    title = title_element.find('span').get_text(strip=True) if title_element else "No title found."

    # 날짜 가져오기
    date_element = soup.find('span', class_='media_end_head_info_datestamp_time _ARTICLE_DATE_TIME')
    date = date_element['data-date-time'] if date_element else "No date found."

    return {
        "title": title,
        "content": content,
        "date": date,
        "url": url
    }


def fetch_daum_content(soup, url):
    """
    Fetches the content from 다음 articles.
    """
    # 본문 내용 가져오기
    content_element = soup.find('div', class_='news_view fs_type1')
    content = content_element.get_text(strip=True) if content_element else "No content found on Daum."

    # # 제목 가져오기
    title_element = soup.find('h3', class_='tit_view', attrs={'data-translation': 'true'})
    title = title_element.get_text(strip=True) if title_element else "No title found."

    # 날짜 가져오기
    date_element = soup.find('span', class_='num_date')
    date = date_element.get_text(strip=True) if date_element else "No date found."

    return {
        "content": content,
        "title": title,
        "date": date,
        "url": url
    }