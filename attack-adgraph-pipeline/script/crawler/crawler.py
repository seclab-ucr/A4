from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import csv

import argparse
import os
import json
from multiprocessing import Pool
import subprocess

from urllib.parse import urlparse


def read_url_list(fpath):
    url_list = []
    with open(fpath, 'r') as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            url_list.append(row['Domain'])
    return url_list


def read_final_domain_set(fpath):
    final_domain_set = set()
    with open(fpath, 'r') as fin:
        data = fin.readlines()
    for row in data:
        row = row.strip()
        original_domain, final_url = row.split(',', 1)
        final_domain = urlparse(final_url)[1]
        final_domain_set.add(final_domain)
    return final_domain_set


def read_original_domain_set(fpath):
    original_domain_set = set()
    with open(fpath, 'r') as fin:
        data = fin.readlines()
    for row in data:
        row = row.strip()
        original_domain, final_url = row.split(',', 1)
        original_domain_set.add(original_domain)
    return original_domain_set


def read_final_domain_to_original_domain_mapping(fpath):
    domain_mapping = {}
    with open(fpath, 'r') as fin:
        data = fin.readlines()
    for row in data:
        row = row.strip()
        original_domain, final_url = row.split(',', 1)
        final_domain = urlparse(final_url)[1]
        domain_mapping[final_domain] = original_domain
    return domain_mapping


def reduce_url_to_domain(url):
    html_filename = url.split('/')[-1]
    html_filename = html_filename.replace('.html', '')
    return html_filename

def func_run_adgraph_api(url, timeline_dir):
    print("Procressing %s" % url)
    cmd = 'python ~/AdGraphAPI/scripts/rules_parser.py --target-dir %s --domain %s' % (
        timeline_dir, url
    )
    os.system(cmd)
    cmd = '~/AdGraphAPI/adgraph ~/rendering_stream/ features/ mappings/ %s parsed_%s' % (url, url)
    os.system(cmd)
    return

def parse_url_to_netloc(url):
    components = urlparse(url)
    netloc = components[1]
    return netloc


parser = argparse.ArgumentParser()
parser.add_argument("--url-list", type=str)
parser.add_argument("--mode", type=str)
parser.add_argument("--crawler-id", type=str, default='0')
parser.add_argument("--concurrency", type=int, default=10)
parser.add_argument("--start-id", type=int, default=0)
parser.add_argument("--end-id", type=int, default=10000)
args = parser.parse_args()

url_list = read_url_list(args.url_list)

HOME_DIR = os.getenv("HOME")

BASE_RENDERING_STREAM_DIR = HOME_DIR + "/rendering_stream"
BASE_HTML_DIR = BASE_RENDERING_STREAM_DIR + "/html"
BASE_TIMELINE_DIR = BASE_RENDERING_STREAM_DIR + "/timeline"
NUM_CRAWLERS = args.concurrency
CRAWLER_SCRIPT_FPATH = HOME_DIR + "/AdGraphAPI/scripts/load_page_adgraph.py"

url_idx = 0

if args.mode == "crawl":
    final_url_to_html_filepath_mapping = {}
    for url in url_list:
        url_idx += 1
        if url_idx % NUM_CRAWLERS == int(args.crawler_id):
            print("Crawling %s" % url)
            cmd = "curl -A %s -Ls -o %s -w %%{url_effective} http://%s" % ('"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3441.0 Safari/537.36"', BASE_HTML_DIR + '/' + url + '.html', url)
            try:
                final_url = subprocess.check_output(cmd, shell=True).decode('ascii')
                final_domain = parse_url_to_netloc(final_url)
            except subprocess.CalledProcessError:
                print("[ERROR] Skipping URL: %s" % url)
                continue
            html_filepath = BASE_HTML_DIR + '/' + url + '.html'
            final_url_to_html_filepath_mapping[final_url] = [final_domain, html_filepath]
        else:
            print("Skipping URL: %s" % url)
            continue
    with open(BASE_RENDERING_STREAM_DIR + '/final_url_to_html_filepath_mapping' + str(args.crawler_id), 'w') as fout:
        for final_url, (final_domain, html_filepath) in final_url_to_html_filepath_mapping.items():
            fout.write(','.join([final_url, final_domain, html_filepath]) + '\n')
if args.mode == "crawl_selenium":
    for url in url_list:
        url_idx += 1
        if url_idx % NUM_CRAWLERS == int(args.crawler_id):
            print("Crawling %s..." % url)
            cmd = "python3 %s --mode no_proxy --domain %s --id %s" % (
                CRAWLER_SCRIPT_FPATH,
                url,
                args.crawler_id
            )
            os.system(cmd)
elif args.mode == "load":
    original_domain_set = read_original_domain_set(HOME_DIR +'/map_local_list_unmod_new.csv')
    for url in url_list:
        url_idx += 1
        if url_idx < args.start_id or url_idx > args.end_id:
            continue
        if url_idx % NUM_CRAWLERS == int(args.crawler_id):
            if url not in original_domain_set:
                continue
            print("Loading %s" % url)
            cmd = 'python3 %s --mode proxy --domain %s --id %s --weirdo' % (
                CRAWLER_SCRIPT_FPATH,
                url, 
                args.crawler_id
            )
            os.system(cmd)
        else:
            print("Skipping URL: %s" % url)
elif args.mode == "run_api":
    final_domain_to_original_domain_mapping = read_final_domain_to_original_domain_mapping(HOME_DIR +'/map_local_list_unmod_new.csv')
    original_domains = list(final_domain_to_original_domain_mapping.values())
    pool = Pool(processes=100)
    for url in original_domains:
        url_idx += 1
        print("URL ID: %i" % url_idx)
        pool.apply_async(func_run_adgraph_api, [url, BASE_TIMELINE_DIR])
    pool.close()
    pool.join()
elif args.mode == "gen_map_local_list_unmodified":
    with open(BASE_RENDERING_STREAM_DIR + '/map_local_list_unmod.csv', 'w') as fout:
        for url in url_list:
            fout.write(','.join([url, BASE_HTML_DIR + '/' + url + '.html']) + '\n')
elif args.mode == "gen_map_local_list_modified":
    with open(BASE_RENDERING_STREAM_DIR + '/map_local_list_mod.csv', 'w') as fout:
        for url in url_list:
            fout.write(','.join([url, BASE_HTML_DIR + '/modified_' + url + '.html']) + '\n')
elif args.mode == "move_timeline_files":
    invalid_cnt = 0
    missing_cnt = 0
    final_domain_set = read_final_domain_set(HOME_DIR + '/map_local_list_unmod_new.csv')
    final_domain_to_original_domain_mapping = read_final_domain_to_original_domain_mapping(HOME_DIR + '/map_local_list_unmod_new.csv')

    all_files_dirs = os.listdir(BASE_RENDERING_STREAM_DIR)
    for file_dir in all_files_dirs:
        if file_dir not in final_domain_set:
            missing_cnt += 1
            continue
        if file_dir in final_domain_set:
            final_domain = file_dir
        original_domain = final_domain_to_original_domain_mapping[final_domain]

        print("Currently processing file: %s" % file_dir)
        json_filenames = os.listdir(BASE_RENDERING_STREAM_DIR + '/' + file_dir)
        if not json_filenames[0].endswith('.json'):
            invalid_cnt += 1
            print("Invalid: %s" % file_dir)
            continue
        cmd = "cp %s %s" % (
            BASE_RENDERING_STREAM_DIR + '/' + file_dir + '/' + json_filenames[0],
            BASE_TIMELINE_DIR + '/' + original_domain + '.json'
        )
        os.system(cmd)
elif args.mode == "rename_timeline_files":
    final_domain_to_original_domain_mapping = read_final_domain_to_original_domain_mapping(HOME_DIR + '/map_local_list_unmod.csv')
    all_timeline_fnames = os.listdir(BASE_TIMELINE_DIR)
    for fname in all_timeline_fnames:
        final_domain = fname.replace('.json', '')
        if final_domain in final_domain_to_original_domain_mapping:
            original_domain = final_domain_to_original_domain_mapping[final_domain]
            cmd = "mv %s %s" % (
                BASE_TIMELINE_DIR + '/' + final_domain + '.json',
                BASE_TIMELINE_DIR + '/' + original_domain + '.json',
            )
            os.system(cmd)
            print("Successfully renamed: %s to %s" % (final_domain, original_domain))
        else:
            print("Non-exsiting final_domain: %s" % final_domain)
            cmd = "rm %s" % (BASE_TIMELINE_DIR + '/' + final_domain + '.json')
            os.system(cmd)

else:
    print("Mode: %s not supported!" % args.mode)
    raise Exception
