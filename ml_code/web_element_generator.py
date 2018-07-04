import app_facing_code as appcode
from PIL import Image
from io import BytesIO
import os as os

script_dir = os.path.dirname(__file__)
image_dir = '../data/element_screenshots'
abs_image_path = os.path.join(script_dir,image_dir)



class WebElementGenerator:


    def __init__(self,driver):
        self.driver = driver



    def generate_elements(self,url):
        
        self.driver.get(url)
        
        all_visible_elements = self.driver.find_elements_by_xpath("//a[not(contains(@style,'display:none'))] | //h2[not(contains(@style,'display:none'))] | //button[not(contains(@style,'display:none'))] | //span[not(contains(@style,'display:none'))] | //input[not(contains(@style,'display:none'))]")


        for element in all_visible_elements:
            
            location = element.location_once_scrolled_into_view
            size = element.size
            x1 = location['x']*2
            y1 = location['y']*2
            x2 = x1 + size['width']*2
            y2 = y1 + size['height']*2

            if x2 != 0 and y2 != 0:
                png = self.driver.get_screenshot_as_png()
                img = Image.open(BytesIO(png))
                img = img.crop((x1,y1,x2,y2))
                img.save('/Users/bardek01/Personal/projects/ml_based_app_testing/data/element_screenshots' +'/' + element.tag_name + '-' + element.text + '.png')

                



if __name__ == '__main__':
    driver = appcode.AppFacing('pc').get_driver()
    web_element_generator = WebElementGenerator(driver)
    web_element_generator.generate_elements("https://www.bbc.co.uk")
    driver.quit()