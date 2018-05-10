# ml_based_app_testing

## Test Case creation based on Machine Learning

The Machine Learning Algorithm is presented an Application [Mobile / Web App] -AUT. Algorithm [Reinforcement Learning - Policy Gradient] starts by performing actions randomly. By training the algorithm on the app, the Algorothim learns to navigate the AUT. A path that is learned by the system is converted into a test case. To be added as part of regression suit.


## Technologies Used

1. Python
2. Selenium - Python
3. Tensorflow
4. Appium


## Running

To run on a pc using Selenium webdriver and chromedriver:

    TARGET_MACHINE=pc CHROMEDRIVER_PATH=`which chromedriver` python ml_code/treasure_hunter.py
