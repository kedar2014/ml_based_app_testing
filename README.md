# ml_based_app_testing

## Test Case creation based on Machine Learning

The Machine Learning Algorithm is presented an Application [Mobile / Web App] -AUT. Algorithm [Reinforcement Learning - Policy Gradient] starts by performing actions randomly. By training the algorithm on the app, the Algorothim learns to navigate the AUT. A path that is learned by the system is converted into a test case. To be added as part of regression suit.


## Technologies Used

1. Python
2. Selenium - Python
3. Tensorflow
4. Appium

# Setup

## Tensorflow

Follow instructions [here](https://www.tensorflow.org/install/install_mac) under `Installing with Virtualenv`

## Appium
```
mkdir appium
cd appium
npm install appium@1.7.1
```

## Running

### Installing project dependencies
    pip install -r requirement.txt --user

#### To run on a pc using Selenium webdriver and chromedriver:

    TARGET_MACHINE=pc CHROMEDRIVER_PATH=`which chromedriver` python ml_code/treasure_hunter.py

#### To run on a mobile 
```
1. Start appium in a terminal. Command: `appium/node_modules/.bin/appium`
2. Connect an android device or start an emulator
3. Run below command    
    TARGET_MACHINE=mobile ADB_DEVICE_ARGS=`adb devices | awk {'print $1'} | sed -n 2p` python ml_code/treasure_hunter.py
```   
