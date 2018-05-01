'''
Created on Apr 24, 2018

@author: M1030443
'''
import pytest
#import os
from elementdetector.pages.iplayer_page import 

@pytest.mark.usefixtures('driver_setup')
class TestImageDetector(): 
    

    def __init__(self):
        self.driver.get('link')

    @pytest.mark.run(order=1)
    def test_visit_url(self):
        self.driver.get('http://bbc.co.uk/iplayer')
    
    def test_closing_cookie(self):
        iplayer_page = IplayerPage(self.driver)
        iplayer_page.close_cookie()

    def get_all_links(self):
        elems = self.driver.find_elements_by_xpath("//a[@href]")
    
    def test_elements_present(self):
        iplayer_page = IplayerPage(self.driver)
        logo_present = iplayer_page.verify_element_present('BBC Header')
        png = self.driver.get_screenshot_as_png()
        iplayer_page.print_element_attributes('BBC Header', png)
        assert logo_present == True
        
#     def test_take_page_screenshot(self):
#         self.driver.save_screenshot('iplayer.png')
        