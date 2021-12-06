import subprocess
import argparse

CONCURRENCY_CRAWL = 70
CONCURRENCY_LOAD = 5


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--mode', type=str)
args = parser.parse_args()

MODE = args.mode


browser_id = 0
if MODE == 'crawl':
    for i in range(CONCURRENCY_CRAWL):
        cmd = "nohup python3 crawler.py --url-list majestic_million.csv --mode crawl_selenium --crawler-id %d --concurrency %d > ../../report/crawl/%d.log &" % (i, CONCURRENCY_CRAWL, i)
        print(cmd)
        subprocess.Popen(cmd, shell=True)
        browser_id += 1
if MODE == 'load':
    for i in range(CONCURRENCY_LOAD):
        cmd = "nohup python3 crawler.py --url-list majestic_million.csv --mode load --crawler-id %d --concurrency %d --start-id 5001 --end-id 10000 > ../../report/load/%d.log &" % (i, CONCURRENCY_LOAD, i)
        print(cmd)
        subprocess.Popen(cmd, shell=True)
        browser_id += 1
