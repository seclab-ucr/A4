import os
import argparse
import time
import logging

from selenium import webdriver
from selenium.common.exceptions import TimeoutException


epoch_time = str(int(time.time() * 10000000))
logging.basicConfig(filename='../report/plt/plt_%s.log' % epoch_time, format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def crawl(
    url,
    driver_path,
    profile_dir,
    proxy_port,
    strategy=None,
    page_load_timeout=120,
    file_write_timeout=5
):
    if strategy == "dist":
        proxy_port += 20
    if strategy == "cent":
        proxy_port = proxy_port
    if strategy == "NA":
        proxy_port = proxy_port
    if strategy is None:
        proxy_port = proxy_port
    
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--disable-application-cache')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--incognito')
    chrome_options.add_argument('--proxy-server=http://localhost:%d' % proxy_port)
    chrome_options.add_argument('--user-data-dir=' + profile_dir)
    chrome_options.add_argument('--user-agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3441.0 Safari/537.36"')

    driver = webdriver.Chrome(driver_path, chrome_options=chrome_options)
    driver.set_page_load_timeout(page_load_timeout)
    
    plt = 0.0

    try:
        print("[Selenium] Loading %s" % "http://" + url)
        driver.get("http://" + url)
        time.sleep(file_write_timeout)
        try:
            navigationStart = driver.execute_script("return window.performance.timing.navigationStart")
            loadEventEnd = driver.execute_script("return window.performance.timing.loadEventEnd")
            plt = loadEventEnd - navigationStart
        except BaseException as ex:
            print('[Main Frame] Something went wrong: ' + str(ex))
            pass
        time.sleep(file_write_timeout)
    except BaseException as ex:
        print('Something went wrong: ' + str(ex))
        pass
    finally:
        driver.quit()

    return plt


parser = argparse.ArgumentParser()
parser.add_argument("--id", type=str, default='1')
parser.add_argument("--plt-list", type=str)
args = parser.parse_args()

HOME_DIR = os.getenv("HOME")

driver_path = HOME_DIR + '/attack-adgraph-pipeline/script/crawler/chromedriver'
base_profile_path = HOME_DIR + '/chrome_profile'

with open(args.plt_list, 'r') as fin:
    data = fin.readlines()
    for row in data:
        row = row.strip()
        if len(row.split(',')) == 3:
            domain, url_id, strategy = row.split(',')
            plt = crawl(domain, driver_path, base_profile_path + args.id, 7777 + int(args.id) % 11, strategy)
            print(plt)
            logger.info(','.join([domain, url_id, strategy, str(plt)]))
        else:
            domain = row
            plt = crawl(domain, driver_path, base_profile_path + args.id, 6666 + int(args.id) % 2, "NA")
            print(plt)
            logger.info(','.join([domain, "original", str(plt)]))
