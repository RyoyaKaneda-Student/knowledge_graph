# coding: UTF-8
import time
from urllib.request import urlopen
from bs4 import BeautifulSoup


def make_soup(url, *, args, logger):
    """
    引数:(str)対象のURL
    返り値:(BeautifulSoup)BeautifulSoupオブジェクト
    サーバーへの負担軽減の為アクセスの間隔を1秒開けます
    """
    time.sleep(1)
    soup = BeautifulSoup(urlopen(url, timeout=3600*5).read().decode('utf-8', 'ignore'), "html.parser")
    if args.output_soap: logger.debug(soup)
    return soup
