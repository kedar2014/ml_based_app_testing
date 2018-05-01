import os
from selenium import webdriver
from PIL import Image
import numpy as np
import io



class AppFacing:
    
    def __init__(self):
       self.counter = 0          
       chromedriver = "/Users/bardek01/Downloads/chromedriver"
       os.environ["webdriver.chrome.driver"] = chromedriver

    #    options = webdriver.ChromeOptions()
    #    options.add_argument('--headless')   
    #    self.driver = webdriver.Chrome(chromedriver,options=options)
       self.driver = webdriver.Chrome(chromedriver)
       
       self.driver.set_window_size(150, 1000)
       self.app = "http://smp-scratch.tools.bbc.co.uk/aimee/machine-learning/treasure-hunt/pages/001.html"
       self.driver.get(self.app)
       self.current_page_url = self.driver.current_url

    def take_current_page_screenshot(self):
        png = self.driver.get_screenshot_as_png()
        img = Image.open(io.BytesIO(png)).convert('1')
        img.save('./temp/result.png')
        return np.array(img).flatten()[:1374400]

    def get_reward(self):
        reward = 0
        if self.driver.current_url==self.current_page_url:
            reward = 0
        elif "bones.html" in self.driver.current_url:
            reward = -2
        elif "treasure.html" in self.driver.current_url:
            reward = 2   
        else: 
            reward = 1

        self.current_page_url = self.driver.current_url    
        return reward

    def get_all_links_on_page(self):
         self.current_page_links = self.driver.find_elements_by_xpath("//a[@href]")
         return self.current_page_links

    def step(self,action_no):
        self.counter = self.counter + 1
        all_links = self.get_all_links_on_page()
        try:
         all_links[action_no].click()
        except:
         print("action not done")

        observation = self.take_current_page_screenshot()
        reward = self.get_reward()

        if self.counter==2:
            done = True
        else:
            done=False


        return observation,reward,done,"info"

                
    def reset(self):
        self.counter=0
        self.driver.get(self.app)
        self.driver.set_window_size(150, 1000)
        observation = self.take_current_page_screenshot()
        return observation