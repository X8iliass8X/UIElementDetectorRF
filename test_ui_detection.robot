*** Settings ***
Library    UIElementDetectorRF   best_Kaggle.pt   

*** Variables ***
${URL}    https://payby.ma/demo/index.php

*** Test Cases ***
Test Button Detection
    ${result}=    Detect Button By Text    ${URL}    Rechercher    0.3    True
    Should Not Be Equal    ${result}    ${None}
    ${xpath}=    Get Element Xpath   ${result}
    Log    Found button at: ${xpath}
    ${confidence}=    Get Detection Confidence    ${result}
    Should Be True    ${confidence} > 0.5


*** Keywords ***
Suite Teardown
    Close Browser