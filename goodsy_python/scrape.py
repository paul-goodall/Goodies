from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

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

    try_count = 1
    while try_count > 0:
        try:
          elems = get_elements(driver, css_str)
        except:
          elems = []
          try_count += 1
          time.sleep(1)
        else:
            if len(elems) > 0:
                elem = elems[0]
                try_count = -1

    return elem

# ==============================================================================














# ==============================================================================
