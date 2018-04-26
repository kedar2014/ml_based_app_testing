'''
Created on Apr 24, 2018

@author: M1030443
'''
from elementdetector.pages.base_page import BasePage

class IplayerPage(BasePage):
    
    bbc_logo = ('css','div.orb-nav-section.orb-nav-blocks')
    user_icon = ('id','idcta-statusbar')
    bbc_header = ('id', 'orb-header')
    
    def verify_element_present(self, element):
        if element == 'BBC Logo':
            return self.is_visible(self.bbc_logo)
        elif element == 'BBC Header':
            return self.is_visible(self.bbc_header)
        else:
            print('wrong element')

    def print_element_attributes(self, element, png):
        if element == 'BBC Logo':
            self.take_element_screenshot(self.bbc_logo, png)
        elif element == 'BBC Header':
            self.take_element_screenshot(self.bbc_header, png)
