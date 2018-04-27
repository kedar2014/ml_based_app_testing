'''
Created on Apr 24, 2018

@author: M1030443
'''
from elementdetector.pages.base_page import BasePage

class IplayerPage(BasePage):
    
    bbc_logo = ('css','div.orb-nav-section.orb-nav-blocks')
    user_icon = ('id','idcta-statusbar')
    bbc_header = ('id', 'orb-header')
    bbc_account_icon = ('id', 'mybbc-wrapper')
    iplayer_logo = ('css', 'li.ipNav__logo')
    home_icon = ('css', 'li.orb-nav-home')
    
    def verify_element_present(self, element):
        if element == 'BBC Logo':
            return self.is_visible(self.bbc_logo)
        elif element == 'BBC Header':
            return self.is_visible(self.bbc_header)
        else:
            print('wrong element')

    def get_elements_from_page(self, png):
        elements = ['BBC_Logo', 'BBC_Header', 'BBC_Account_Icon', 'iPlayer_Logo', 'Home_Icon']
        for element in elements:
            locator = ''
            if element == 'BBC_Logo':
                locator = self.bbc_logo
            elif element == 'BBC_Header':
                locator = self.bbc_header
            elif element == 'BBC_Account_Icon':
                locator = self.bbc_account_icon
            elif element == 'iPlayer_Logo':
                locator = self.iplayer_logo
            elif element == 'Home_Icon':
                locator = self.home_icon
            self.take_element_screenshot(element, locator, png)
