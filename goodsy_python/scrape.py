from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time

# ==============================================================================

def get_chrome_driver():
    service = Service()
    options = webdriver.ChromeOptions()
    driver  = webdriver.Chrome(service=service, options=options)
    return driver

# ==============================================================================

def get_page_source(driver):
    return driver.page_source

# ==============================================================================

def get_elements(driver, searchstr, eltype='css selector'):
    # this defaults to css string just like in JQUERY
    # options for eltype:
    # ID = "id"
    # NAME = "name"
    # XPATH = "xpath"
    # LINK_TEXT = "link text"
    # PARTIAL_LINK_TEXT = "partial link text"
    # TAG_NAME = "tag name"
    # CLASS_NAME = "class name"
    # CSS_SELECTOR = "css selector"

    return driver.find_elements(eltype, searchstr)

# ==============================================================================

def get_element_keep_trying(driver,css_str):

    max_tries = 30
    try_count = 1
    while try_count < max_tries:
        print(f'Trying [{css_str}] - [{try_count}]')
        elem = ''
        try:
          elem = get_elements(driver, css_str)[0]
        except IndexError:
          try_count += 1
          time.sleep(1)
        else:
            max_tries = -1

    return elem

# ==============================================================================

def wait_until_url_contains_str(driver,str):

    try_count = 1
    while try_count > 0:
        print(f'Waiting for URL to contain [{str}] - [{try_count}]')
        if str in driver.current_url:
            try_count = -1
        else:
            try_count += 1
            time.sleep(1)


# ==============================================================================











# ==============================================================================
