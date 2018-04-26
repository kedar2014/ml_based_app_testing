import pytest
from appium import webdriver

@pytest.fixture(scope='class')
def driver_setup(request):
    capabilities = {
        'platformName': 'Android',
        'udid': '0aef21ee02e4221a',
        'browserName': 'chrome',
        'deviceName': 'Nexus 5'
    }
    url = 'http://localhost:4723/wd/hub'
    driver = webdriver.Remote(url, capabilities)
    
    if request.cls is not None:
        request.cls.driver = driver
    
    yield driver
    print('teardown code')
    driver.quit()