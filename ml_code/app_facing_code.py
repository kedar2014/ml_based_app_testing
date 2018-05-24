import os
from selenium import webdriver
from PIL import Image
import numpy as np
import io
import os
from appium_helper import AppiumHelper
#from appium import webdriver


class AppFacing:

    def __init__(self,device_type,render):
       self.counter = 0
       if device_type=="pc":
        chromedriver = os.environ.get('CHROMEDRIVER_PATH')

        if chromedriver is None:
            raise ValueError('Please set CHROMEDRIVER_PATH environment variable')

        os.environ["webdriver.chrome.driver"] = chromedriver
        options = webdriver.ChromeOptions()
        options.add_argument("disable-infobars")
        if render==True:
           options.add_argument("--headless")
           options.add_argument("window-size=600x400")
        
        self.driver = webdriver.Chrome(chromedriver,options=options)
        self.size = self.driver.get_window_size()
        # self.size['width'] = int(self.size['width']*0.25)
        # self.size['height'] = int(self.size['height']*0.80)
        # self.size['width'] = 600
        # self.size['height'] = 400

        self.driver.set_window_size(self.size['width'], self.size['height'])

       elif device_type=='mobile':
        capabilities = AppiumHelper.get_device_capabilities()
        url = 'http://localhost:4723/wd/hub'
        self.driver = webdriver.Remote(url, capabilities)

       self.app = "http://smp-scratch.tools.bbc.co.uk/aimee/machine-learning/treasure-hunt/pages/001.html"
       self.driver.get(self.app)
       self.current_page_url = self.driver.current_url

    def get_observation_size(self):
        return self.width, self.height

    def take_current_page_screenshot(self):
        png = self.driver.get_screenshot_as_png()
        img = Image.open(io.BytesIO(png)).convert('1')
        img.save('/tmp/result.png')
        self.width,self.height = img.size
        return np.array(img).flatten()[:(self.width * self.height)]

    def get_reward(self):
        reward = 0
        if self.driver.current_url==self.current_page_url:
            reward = 0
        elif "bones.html" in self.driver.current_url:
            reward = -1
        elif "treasure.html" in self.driver.current_url:
            reward = 1
        else:
            reward = 0
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
         reward = self.get_reward()
        except:
         reward = -1
         print("action not done")
        observation = self.take_current_page_screenshot()


        if self.counter==2:
            done = True
        else:
            done=False

        return observation,reward,done,"info"

    def reset(self):
        self.counter=0
        self.driver.get(self.app)
        #self.driver.set_window_size(150, 1000)
        observation = self.take_current_page_screenshot()
        return observation

    def resize_window(self):
        self.driver.execute_script("document.body.style.zoom='.4'")
