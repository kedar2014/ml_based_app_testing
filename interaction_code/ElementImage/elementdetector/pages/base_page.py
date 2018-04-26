'''
Created on Apr 24, 2018

@author: M1030443
'''
from elementdetector.screen import Screen

class BasePage(Screen):
    
    continue_link = ('id','bbccookies-continue-button')
        
    def close_cookie(self):
        self.click(self.continue_link)