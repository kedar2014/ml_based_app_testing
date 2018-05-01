from time import sleep

from selenium.common.exceptions import NoSuchElementException, WebDriverException
from appium.webdriver.common.touch_action import TouchAction
from pip._vendor.distlib.locators import Locator
from PIL import Image
from io import BytesIO
import os

class Screen:

    def __init__(self, driver):
        self.driver = driver

    # get elements
    def get_element(self, locator):
        """
        Returns element based on provided locator.

        Locator include the method and locator value in a tuple.
        :param locator:
        :return:
        """

        method = locator[0]
        values = locator[1]

        if type(values) is str:
            return self.get_element_by_type(method, values)
        elif type(values) is list:
            for value in values:
                try:
                    return self.get_element_by_type(method, value)
                except NoSuchElementException:
                    pass
            raise NoSuchElementException

    def get_element_by_type(self, method, value):
        if method == 'accessibility_id':
            return self.driver.find_element_by_accessibility_id(value)
        elif method == 'android':
            return self.driver.find_element_by_android_uiautomator('new UiSelector().%s' % value)
        elif method == 'ios':
            return self.driver.find_element_by_ios_uiautomation(value)
        elif method == 'class_name':
            return self.driver.find_element_by_class_name(value)
        elif method == 'id':
            return self.driver.find_element_by_id(value)
        elif method == 'xpath':
            return self.driver.find_element_by_xpath(value)
        elif method == 'name':
            return self.driver.find_element_by_name(value)
        elif method == 'css':
            return self.driver.find_element_by_css_selector(value)
        else:
            raise Exception('Invalid locator method.')

    def get_elements(self, locator):
        """
        Returns element based on provided locator.

        Locator include the method and locator value in a tuple.
        :param locator:
        :return:
        """

        method = locator[0]
        values = locator[1]

        if type(values) is str:
            return self.get_elements_by_type(method, values)
        elif type(values) is list:
            for value in values:
                try:
                    return self.get_elements_by_type(method, value)
                except NoSuchElementException:
                    pass
            raise NoSuchElementException

    def get_elements_by_type(self, method, value):
        if method == 'accessibility_id':
            return self.driver.find_elements_by_accessibility_id(value)
        elif method == 'android':
            return self.driver.find_elements_by_android_uiautomator(value)
        elif method == 'ios':
            return self.driver.find_elements_by_ios_uiautomation(value)
        elif method == 'class_name':
            return self.driver.find_elements_by_class_name(value)
        elif method == 'id':
            return self.driver.find_elements_by_id(value)
        elif method == 'xpath':
            return self.driver.find_elements_by_xpath(value)
        elif method == 'name':
            return self.driver.find_elements_by_name(value)
        elif method == 'css':
            return self.driver.find_elements_by_css_selector(value)
        else:
            raise Exception('Invalid locator method.')

    # element visible
    def is_visible(self, locator):
        try:
            self.get_element(locator).is_displayed()
            return True
        except NoSuchElementException:
            return False

    # element present
    def is_present(self, locator):
        try:
            self.get_element(locator)
            return True
        except NoSuchElementException:
            return False

    # waits
    def wait_visible(self, locator, timeout=10):
        i = 0
        while i != timeout:
            try:
                self.is_visible(locator)
                return self.get_element(locator)
            except NoSuchElementException:
                sleep(1)
                i += 1
        raise Exception('Element never became visible: %s (%s)' % (locator[0], locator[1]))

    def wait_for_text(self, locator, text, timeout=10):
        i = 0
        while i != timeout:
            try:
                element = self.get_element(locator)
                element_text = element.text
                if element_text.lower() == text.lower():
                    return True
                else:
                    pass
            except NoSuchElementException:
                pass
            sleep(1)
            i += 1
        raise Exception('Element text never became visible: %s (%s) - %s' % (locator[0], locator[1], text))

    # clicks and taps
    def click(self, locator):
        element = self.wait_visible(locator)
        element.click()

    # send keys
    def send_keys(self, locator, text):
        element = self.wait_visible(locator)
        element.send_keys(text)

    # gestures
    def swipe_to_element(self, scrollable_element_locator, target_element_locator, direction, duration=None):
        scrollable_element_attributes = self.get_element_attributes(scrollable_element_locator)
        limit = 5
        attempts = 0
        while True:
            if attempts == limit:
                raise Exception('Could not swipe to element')
            if self.is_visible(target_element_locator):
                break
            else:
                if direction == 'up':
                    self.driver.swipe(
                        scrollable_element_attributes['center_x'],
                        scrollable_element_attributes['top'] + 1,
                        scrollable_element_attributes['center_x'],
                        scrollable_element_attributes['bottom'] - 1,
                        duration
                    )
                elif direction == 'down':
                    self.driver.swipe(
                        scrollable_element_attributes['center_x'],
                        scrollable_element_attributes['bottom'] - 1,
                        scrollable_element_attributes['center_x'],
                        scrollable_element_attributes['top'] + 1,
                        duration
                    )
                elif direction == 'left':
                    self.driver.swipe(
                        scrollable_element_attributes['left'] + 1,
                        scrollable_element_attributes['center_y'],
                        scrollable_element_attributes['right'] - 1,
                        scrollable_element_attributes['center_y'],
                        duration
                    )
                elif direction == 'right':
                    self.driver.swipe(
                        scrollable_element_attributes['right'] - 1,
                        scrollable_element_attributes['center_y'],
                        scrollable_element_attributes['left'] + 1,
                        scrollable_element_attributes['center_y'],
                        duration
                    )
                else:
                    raise Exception('Invalid direction value: %s' % direction)
            attempts += 1

    def long_press(self, locator, duration=1000):
        element = self.get_element(locator)
        action = TouchAction(self.driver)
        action.long_press(element, None, None, duration).perform()

    def get_element_attributes(self, locator):
        element = self.get_element(locator)
        return {
            'x': element.location['x'],
            'y': element.location['y'],
            'bottom': element.location['y'] + element.size['height'],
            'right': element.location['x'] + element.size['width'],
            'center_x': (element.size['width']/2) + element.location['x'],
            'center_y': (element.size['height']/2) + element.location['y'],
            'height': element.size['height'],
            'width': element.size['width']
        }

    def pull_to_refresh(self, locator, duration=1000):
        scrollable_element_attributes = self.get_element_attributes(locator)
        self.driver.swipe(
            scrollable_element_attributes['center_x'],
            scrollable_element_attributes['top'] + 1,
            scrollable_element_attributes['center_x'],
            scrollable_element_attributes['bottom'] - 1,
            duration
        )

    def hide_keyboard(self):
        try:
            sleep(1)
            self.driver.hide_keyboard()
        except WebDriverException:
            pass

    def get_device_size(self):
        webview = self.driver.contexts[1]
        self.driver.switch_to.context('NATIVE_APP')
        device_size = self.driver.get_window_size()
        self.driver.switch_to.context(webview)
        return device_size


    #take screenshot of element
    def take_element_screenshot(self, element_name, locator, png):
        attributes = self.get_element_attributes(locator)
        device_size = self.get_device_size()
        print('*******')
        print(str(attributes['x']) + ':' + str(attributes['y']) + ':' + str(attributes['right']) + ':' +  str(attributes['bottom']) + ':' +  str(attributes['height']) + ':' +  str(attributes['width']))
        print('*******')
        x, y, width, height = [attributes['x'], attributes['y'], attributes['right'], attributes['bottom']]
        im = Image.open(BytesIO(png)) # uses PIL library to open image in memory
        im = im.crop((3*x, 3*y, 2.5*width, 3*height)) # defines crop points
        screenshot_name = element_name + '.png'
        im.save("screenshots/" + screenshot_name) # saves new cropped image
