#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created at daytime July 23 with a cup of coffee
@author: Akihiro Inui
"""
# Run like this e.g. collect banana images for 100 and save them in data/banana directory
# python image_collector -k banana -o ./data -n 100

import bs4
from sys import argv
import urllib.request, urllib.error
import os
import argparse
import sys
import json


def get_soup(url, header):
    # Use BeautifulSoup for scraper
    return bs4.BeautifulSoup(urllib.request.urlopen(urllib.request.Request(url, headers=header)), 'html.parser')


def main(args):
    # Arg parser
    # -k: Keyword for image search. Several keywords can work like; banana+monkey
    # -n: Number of image files to download
    # -o: Output directory
    parser = argparse.ArgumentParser(description='Options for scraping Google images')
    parser.add_argument('-k', '--search', default='banana', type=str, help='Search keywords')
    parser.add_argument('-n', '--num_images', default=200, type=int, help='Number of images to scrape')
    parser.add_argument('-o', '--directory', default='./', type=str, help='Output directory')
    args = parser.parse_args()

    # Put several keywords together
    query = args.search.split()
    query = '+'.join(query)

    # Make output folder if it does not exist
    save_directory = args.directory + '/' + query
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Define scraper
    url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(query, encoding='utf-8') + "&source=lnms&tbm=isch"
    header = {'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3)"
                            "AppleWebKit/537.36 (KHTML, like Gecko)"
                            "Chrome/43.0.2357.134 Safari/537.36"}
    soup = get_soup(url, header)

    image_data_list = []
    # Get search result, (image url and extension)
    for a in soup.find_all("div", {"class": "rg_meta"}):
        link, img_type = json.loads(a.text)["ou"], json.loads(a.text)["ity"]
        image_data_list.append((link, img_type))

    # Save each image file
    for index, (img, img_type) in enumerate(image_data_list):
        try:
            img_type = img_type if len(img_type) > 0 else 'jpg'
            print("Downloading image {} ({}), type is {}".format(index, img, img_type))
            raw_img = urllib.request.urlopen(img).read()
            f = open(os.path.join(save_directory, "img_"+str(index)+"."+img_type), 'wb')
            f.write(raw_img)
            f.close()
        except Exception as e:
            print("could not load : "+img)
            print(e)


if __name__ == "__main__":
    try:
        main(argv)
    except KeyboardInterrupt:
        pass
    sys.exit()
