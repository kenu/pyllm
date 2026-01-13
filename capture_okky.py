from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os
from datetime import datetime
import time

def capture_okky():
    desktop_path = os.path.expanduser("~/Desktop")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_path = os.path.join(desktop_path, f"okky_screenshot_{timestamp}.png")
    
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--window-size=1920,1080')
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.get('http://okky.kr')
    time.sleep(3)
    driver.save_screenshot(screenshot_path)
    driver.quit()
    
    print(f"Screenshot saved: {screenshot_path}")
    return screenshot_path

if __name__ == "__main__":
    capture_okky()
