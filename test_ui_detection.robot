*** Settings ***
Library    UIElementDetectorRF   best_Kaggle.pt   

*** Variables ***
${URL}    https://payby.ma/demo/index.php


*** Test Cases ***
Test Dropdown Selection
    ${dropdown}=    Detect Dropdown By Text    
    ...    https://formstone.it/components/dropdown/demo    
    ...    option_text=One  
    ...    label_text=Label    
    ...    confidence_threshold=0.7
    Should Not Be Equal    ${dropdown}    ${None}
    ${dropdown_xpath}=    Get Element XPath    ${dropdown}
    Log    message: ${dropdown_xpath}


*** Keywords ***
Suite Teardown
    Close Browser