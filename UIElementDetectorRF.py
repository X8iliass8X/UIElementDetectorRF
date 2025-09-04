import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict, Optional
import time
import os
from dataclasses import dataclass
from robot.api import logger


@dataclass
class DetectionResult:
    """Class to store detection results"""
    xpath: str
    dom_element: dict
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    screenshot_path: str
    annotated_screenshot_path: str


class UIElementDetectorRF:
    """
    Robot Framework library for detecting UI elements using YOLO and returning their XPath/DOM information.
    
    This library provides keywords for automated web element detection using computer vision.
    """
    
    ROBOT_LIBRARY_SCOPE = 'GLOBAL'
    ROBOT_LIBRARY_VERSION = '1.0'
    
    def __init__(self, model_path: str, chromedriver_path: str = None):
        """
        Initialize the UI Element Detector Robot Framework library.
        
        Arguments:
        - model_path: Path to the trained YOLO model file
        - chromedriver_path: Path to chromedriver executable (optional)
        
        Example:
        | Library | UIElementDetectorRF | /path/to/model.pt | /path/to/chromedriver |
        """
        logger.info(f"Initializing UIElementDetectorRF with model: {model_path}")
        self.model = YOLO(model_path)
        self.driver = None
        self.chromedriver_path = chromedriver_path
        self.screenshot_counter = 0
        
        # Create output directory for screenshots
        os.makedirs("screenshots", exist_ok=True)
        
    def _setup_driver(self):
        """Setup Chrome WebDriver with English language preference"""
        if self.driver is None:
            chrome_options = Options()
            chrome_options.add_argument("--lang=en-US")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_experimental_option("prefs", {
                "intl.accept_languages": "en-US,en"
            })
        
            if self.chromedriver_path:
                service = Service(self.chromedriver_path)
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
            else:
                self.driver = webdriver.Chrome(options=chrome_options)

    def _take_screenshot(self) -> Tuple[np.ndarray, str]:
        """Take a screenshot and return as numpy array and file path"""
        self.screenshot_counter += 1
        screenshot_path = f"screenshots/screenshot_{self.screenshot_counter}.png"
        
        self.driver.save_screenshot(screenshot_path)
        img = cv2.imread(screenshot_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img_rgb, screenshot_path

    def _get_page_dimensions(self) -> Tuple[int, int, int]:
        """Get page dimensions and viewport height"""
        dimensions = self.driver.execute_script("""
            return {
                pageHeight: Math.max(document.body.scrollHeight, document.documentElement.scrollHeight),
                viewportHeight: window.innerHeight,
                currentScrollY: window.pageYOffset || document.documentElement.scrollTop
            };
        """)
        return dimensions['pageHeight'], dimensions['viewportHeight'], dimensions['currentScrollY']

    def _scroll_to_position(self, y_position: int):
        """Scroll to a specific Y position on the page"""
        self.driver.execute_script(f"window.scrollTo(0, {y_position});")
        time.sleep(0.5)

    def _take_full_page_screenshots(self, overlap_ratio: float = 0.1) -> List[Tuple[np.ndarray, str, int]]:
        """Take multiple screenshots to cover the entire page"""
        page_height, viewport_height, _ = self._get_page_dimensions()
        
        if page_height <= viewport_height:
            # Page fits in one viewport
            self._scroll_to_position(0)
            img, screenshot_path = self._take_screenshot()
            return [(img, screenshot_path, 0)]
        
        step_size = int(viewport_height * (1 - overlap_ratio))
        screenshots = []
        current_y = 0
        
        while current_y < page_height:
            self._scroll_to_position(current_y)
            img, screenshot_path = self._take_screenshot()
            screenshots.append((img, screenshot_path, current_y))
            
            current_y += step_size
            
            # Handle the last screenshot
            if current_y >= page_height and screenshots:
                last_scroll = screenshots[-1][2]
                if page_height - last_scroll > viewport_height * 0.5:
                    # Take final screenshot at bottom
                    final_y = max(0, page_height - viewport_height)
                    if final_y != last_scroll:
                        self._scroll_to_position(final_y)
                        img, screenshot_path = self._take_screenshot()
                        screenshots.append((img, screenshot_path, final_y))
                break
        
        self._scroll_to_position(0)
        return screenshots

    def _adjust_bbox_for_scroll(self, bbox: Tuple[int, int, int, int], scroll_offset: int) -> Tuple[int, int, int, int]:
        """Adjust bounding box coordinates to account for page scroll position"""
        x1, y1, x2, y2 = bbox
        return (x1, y1 + scroll_offset, x2, y2 + scroll_offset)

    def _run_yolo_detection(self, image: np.ndarray, class_filter: List[str] = None) -> List[Dict]:
        """Run YOLO detection on image"""
        results = self.model(image)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    
                    if class_filter and class_name not in class_filter:
                        continue
                    
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy().astype(int)
                    
                    detections.append({
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': tuple(bbox),
                    })
        
        return detections

    def _get_elements_in_bbox(self, bbox: Tuple[int, int, int, int]) -> List[dict]:
        """Get all DOM elements within the bounding box"""
        x1, y1, x2, y2 = map(int, bbox)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
    
        element = self.driver.execute_script(
            f"return document.elementFromPoint({center_x}, {center_y});"
        )
    
        if not element:
            return []
    
        elements = self.driver.execute_script("""
            var x1 = arguments[1];
            var y1 = arguments[2];
            var x2 = arguments[3];
            var y2 = arguments[4];
            var elements = [];
            var allElements = document.querySelectorAll('*');
        
            for (var i = 0; i < allElements.length; i++) {
                var elemRect = allElements[i].getBoundingClientRect();
                if (elemRect.left < x2 && elemRect.right > x1 && 
                    elemRect.top < y2 && elemRect.bottom > y1) {
                    elements.push({
                        element: allElements[i],
                        rect: {
                            left: elemRect.left,
                            top: elemRect.top,
                            right: elemRect.right,
                            bottom: elemRect.bottom
                        }
                    });
                }
            }
            return elements;
        """, element, x1, y1, x2, y2)
    
        return elements

    def _get_elements_in_bbox_with_scroll(self, bbox: Tuple[int, int, int, int], scroll_offset: int) -> List[dict]:
        """Get all DOM elements within the bounding box, accounting for scroll position"""
        x1, y1, x2, y2 = bbox
        viewport_y1 = y1 - scroll_offset
        viewport_y2 = y2 - scroll_offset
        
        # Check if element is visible in current viewport
        viewport_height = self.driver.execute_script("return window.innerHeight;")
        current_scroll = self.driver.execute_script("return window.pageYOffset || document.documentElement.scrollTop;")
        
        if viewport_y2 < 0 or viewport_y1 > viewport_height:
            # Scroll to make element visible
            scroll_to = max(0, y1 - viewport_height // 2)
            self._scroll_to_position(scroll_to)
            time.sleep(0.3)
            
            # Recalculate viewport coordinates
            new_scroll = self.driver.execute_script("return window.pageYOffset || document.documentElement.scrollTop;")
            viewport_y1 = y1 - new_scroll
            viewport_y2 = y2 - new_scroll
        
        return self._get_elements_in_bbox((x1, viewport_y1, x2, viewport_y2))

    def _get_xpath(self, element) -> str:
        """Generate XPath for an element"""
        return self.driver.execute_script("""
            function getXPath(element) {
                if (element.id !== '') {
                    return '//*[@id="' + element.id + '"]';
                }
                if (element === document.body) {
                    return '/html/body';
                }
                
                var ix = 0;
                var siblings = element.parentNode.childNodes;
                for (var i = 0; i < siblings.length; i++) {
                    var sibling = siblings[i];
                    if (sibling === element) {
                        return getXPath(element.parentNode) + '/' + element.tagName.toLowerCase() + '[' + (ix + 1) + ']';
                    }
                    if (sibling.nodeType === 1 && sibling.tagName === element.tagName) {
                        ix++;
                    }
                }
            }
            return getXPath(arguments[0]);
        """, element)
    
    def _get_element_info(self, element) -> dict:
        """Get comprehensive information about an element"""
        return self.driver.execute_script("""
            var elem = arguments[0];
            return {
                tagName: elem.tagName,
                id: elem.id,
                className: elem.className,
                text: elem.textContent || elem.innerText || '',
                value: elem.value || '',
                placeholder: elem.placeholder || '',
                alt: elem.alt || '',
                title: elem.title || '',
                href: elem.href || '',
                src: elem.src || '',
                type: elem.type || '',
                name: elem.name || '',
                checked: elem.checked || false,
                selected: elem.selected || false,
                attributes: Array.from(elem.attributes).map(attr => ({
                    name: attr.name,
                    value: attr.value
                })),
                rect: elem.getBoundingClientRect()
            };
        """, element)

    def _create_annotated_screenshot(self, original_path: str, detections: List[Dict], 
                                   target_detection: Dict = None) -> str:
        """Create an annotated screenshot with bounding boxes"""
        img = Image.open(original_path)
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            color = 'red' if detection == target_detection else 'blue'
            width = 3 if detection == target_detection else 2
            
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            label = f"{detection['class_name']} ({detection['confidence']:.2f})"
            draw.text((x1, max(0, y1-20)), label, fill=color, font=font)
        
        annotated_path = original_path.replace('.png', '_annotated.png')
        img.save(annotated_path)
        return annotated_path

    def _create_full_page_annotated_screenshot(self, screenshots: List[Tuple[np.ndarray, str, int]], 
                                             all_detections: List[Dict], target_detection: Dict = None) -> str:
        """Create an annotated screenshot combining all page sections"""
        if not screenshots:
            return ""
        
        # Calculate dimensions
        total_height = sum(img.shape[0] for img, _, _ in screenshots)
        max_width = max(img.shape[1] for img, _, _ in screenshots)
        
        # Create combined image
        combined_img = Image.new('RGB', (max_width, total_height))
        
        current_y = 0
        for img, _, _ in screenshots:
            pil_img = Image.fromarray(img)
            combined_img.paste(pil_img, (0, current_y))
            current_y += pil_img.height
        
        # Draw annotations
        draw = ImageDraw.Draw(combined_img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        for detection in all_detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            color = 'red' if detection == target_detection else 'blue'
            width = 3 if detection == target_detection else 2
            
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            label = f"{detection['class_name']} ({detection['confidence']:.2f})"
            draw.text((x1, max(0, y1-20)), label, fill=color, font=font)
        
        annotated_path = f"screenshots/full_page_annotated_{self.screenshot_counter}.png"
        combined_img.save(annotated_path)
        return annotated_path
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity"""
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        if text1 == text2:
            return 1.0
        
        if text1 in text2 or text2 in text1:
            return 0.8
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

    def _calculate_distance(self, rect1: dict, rect2: dict) -> float:
        """Calculate distance between two rectangles"""
        center1_x = (rect1['left'] + rect1['right']) / 2
        center1_y = (rect1['top'] + rect1['bottom']) / 2
        center2_x = (rect2['left'] + rect2['right']) / 2
        center2_y = (rect2['top'] + rect2['bottom']) / 2
        
        return ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5

    # Robot Framework Keywords below

    def detect_button_by_text(self, url: str, button_text: str, 
                            confidence_threshold: str = "0.5", scan_full_page: str = "True") -> dict:
        """Detect a button by its text content.
        
        Arguments:
        - url: The URL to navigate to
        - button_text: The text to search for in buttons
        - confidence_threshold: Minimum confidence threshold (default: 0.5)
        - scan_full_page: Whether to scan the full page or just visible area (default: True)
        
        Returns a dictionary with detection results or None if not found.
        
        Example:
        | ${result}= | Detect Button By Text | https://example.com | Submit | 0.7 | True |
        | Should Not Be Equal | ${result} | ${None} |
        | Log | Found button at xpath: ${result.xpath} |
        """
        logger.info(f"Detecting button with text '{button_text}' on {url}")
        confidence_threshold = float(confidence_threshold)
        scan_full_page = scan_full_page.lower() == 'true'
        
        self._setup_driver()
        self.driver.get(url)
        time.sleep(2)
        
        if scan_full_page:
            screenshots = self._take_full_page_screenshots()
            all_detections = []
            
            for img, screenshot_path, scroll_offset in screenshots:
                detections = self._run_yolo_detection(img, class_filter=['button'])
                for detection in detections:
                    detection['bbox'] = self._adjust_bbox_for_scroll(detection['bbox'], scroll_offset)
                    detection['scroll_offset'] = scroll_offset
                    detection['screenshot_path'] = screenshot_path
                all_detections.extend(detections)
        else:
            img, screenshot_path = self._take_screenshot()
            all_detections = self._run_yolo_detection(img, class_filter=['button'])
            for detection in all_detections:
                detection['scroll_offset'] = 0
                detection['screenshot_path'] = screenshot_path
        
        best_match = None
        best_score = 0
        
        for detection in all_detections:
            if detection['confidence'] < confidence_threshold:
                continue
            
            bbox = detection['bbox']
            scroll_offset = detection.get('scroll_offset', 0)
            
            if scan_full_page:
                elements = self._get_elements_in_bbox_with_scroll(bbox, scroll_offset)
            else:
                elements = self._get_elements_in_bbox(bbox)
            
            for elem_data in elements:
                element = elem_data['element']
                elem_info = self._get_element_info(element)
                
                if elem_info['tagName'].lower() in ['button', 'input'] or \
                   'button' in elem_info['className'].lower():
                    
                    element_text = elem_info['text'] or elem_info['value']
                    similarity = self._text_similarity(button_text, element_text)
                    combined_score = detection['confidence'] * similarity
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        xpath = self._get_xpath(element)
                        
                        if scan_full_page:
                            annotated_path = self._create_full_page_annotated_screenshot(
                                screenshots, all_detections, detection
                            )
                        else:
                            annotated_path = self._create_annotated_screenshot(
                                detection['screenshot_path'], all_detections, detection
                            )
                        
                        best_match = {
                            'xpath': xpath,
                            'dom_element': elem_info,
                            'confidence': detection['confidence'],
                            'bbox': bbox,
                            'screenshot_path': detection['screenshot_path'],
                            'annotated_screenshot_path': annotated_path
                        }
        
        if best_match:
            logger.info(f"Button found with confidence {best_match['confidence']:.2f} at xpath: {best_match['xpath']}")
        else:
            logger.warn(f"Button with text '{button_text}' not found")
        
        return best_match
    
    def detect_input_by_label(self, url: str, label_text: str, 
                            confidence_threshold: str = "0.5", scan_full_page: str = "True") -> dict:
        """Detect an input field by its associated label.
        
        Arguments:
        - url: The URL to navigate to
        - label_text: The label text to search for
        - confidence_threshold: Minimum confidence threshold (default: 0.5)
        - scan_full_page: Whether to scan the full page or just visible area (default: True)
        
        Returns a dictionary with detection results or None if not found.
        
        Example:
        | ${result}= | Detect Input By Label | https://example.com | Email Address | 0.6 |
        | Should Not Be Equal | ${result} | ${None} |
        """
        logger.info(f"Detecting input field with label '{label_text}' on {url}")
        confidence_threshold = float(confidence_threshold)
        scan_full_page = scan_full_page.lower() == 'true'
        
        self._setup_driver()
        self.driver.get(url)
        time.sleep(2)
        
        if scan_full_page:
            screenshots = self._take_full_page_screenshots()
            all_detections = []
            
            for img, screenshot_path, scroll_offset in screenshots:
                detections = self._run_yolo_detection(img, class_filter=['input', 'textbox'])
                for detection in detections:
                    detection['bbox'] = self._adjust_bbox_for_scroll(detection['bbox'], scroll_offset)
                    detection['scroll_offset'] = scroll_offset
                    detection['screenshot_path'] = screenshot_path
                all_detections.extend(detections)
        else:
            img, screenshot_path = self._take_screenshot()
            all_detections = self._run_yolo_detection(img, class_filter=['input', 'textbox'])
            for detection in all_detections:
                detection['scroll_offset'] = 0
                detection['screenshot_path'] = screenshot_path
        
        best_match = None
        best_score = 0
        
        for detection in all_detections:
            if detection['confidence'] < confidence_threshold:
                continue
            
            bbox = detection['bbox']
            scroll_offset = detection.get('scroll_offset', 0)
            
            if scan_full_page:
                elements = self._get_elements_in_bbox_with_scroll(bbox, scroll_offset)
            else:
                elements = self._get_elements_in_bbox(bbox)
            
            for elem_data in elements:
                element = elem_data['element']
                elem_info = self._get_element_info(element)
                
                if elem_info['tagName'].lower() in ['input', 'textarea', 'select']:
                    max_similarity = 0
                    
                    # Method 1: Label with 'for' attribute
                    if elem_info['id']:
                        try:
                            label_element = self.driver.find_element(
                                By.CSS_SELECTOR, f"label[for='{elem_info['id']}']"
                            )
                            label_info = self._get_element_info(label_element)
                            similarity = self._text_similarity(label_text, label_info['text'])
                            if similarity > max_similarity:
                                max_similarity = similarity
                        except:
                            pass
                    
                    # Method 2: Parent label
                    try:
                        parent_label = self.driver.execute_script("""
                            var elem = arguments[0];
                            var parent = elem.parentElement;
                            while (parent && parent.tagName.toLowerCase() !== 'label') {
                                parent = parent.parentElement;
                            }
                            return parent;
                        """, element)
                        
                        if parent_label:
                            label_info = self._get_element_info(parent_label)
                            similarity = self._text_similarity(label_text, label_info['text'])
                            if similarity > max_similarity:
                                max_similarity = similarity
                    except:
                        pass
                    
                    # Method 3: Nearby labels
                    try:
                        nearby_labels = self.driver.execute_script("""
                            var elem = arguments[0];
                            var labels = [];
                            var allLabels = document.querySelectorAll('label');
                            
                            for (var i = 0; i < allLabels.length; i++) {
                                var labelRect = allLabels[i].getBoundingClientRect();
                                var elemRect = elem.getBoundingClientRect();
                                
                                var distance = Math.abs(labelRect.bottom - elemRect.top) + 
                                             Math.abs(labelRect.left - elemRect.left);
                                
                                if (distance < 100) {
                                    labels.push(allLabels[i]);
                                }
                            }
                            return labels;
                        """, element)
                        
                        for nearby_label in nearby_labels:
                            label_info = self._get_element_info(nearby_label)
                            similarity = self._text_similarity(label_text, label_info['text'])
                            if similarity > max_similarity:
                                max_similarity = similarity
                    except:
                        pass
                    
                    # Check placeholder text
                    if elem_info['placeholder']:
                        similarity = self._text_similarity(label_text, elem_info['placeholder'])
                        if similarity > max_similarity:
                            max_similarity = similarity
                    
                    combined_score = detection['confidence'] * max_similarity
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        xpath = self._get_xpath(element)
                        
                        if scan_full_page:
                            annotated_path = self._create_full_page_annotated_screenshot(
                                screenshots, all_detections, detection
                            )
                        else:
                            annotated_path = self._create_annotated_screenshot(
                                detection['screenshot_path'], all_detections, detection
                            )
                        
                        best_match = {
                            'xpath': xpath,
                            'dom_element': elem_info,
                            'confidence': detection['confidence'],
                            'bbox': bbox,
                            'screenshot_path': detection['screenshot_path'],
                            'annotated_screenshot_path': annotated_path
                        }
        
        if best_match:
            logger.info(f"Input field found with confidence {best_match['confidence']:.2f} at xpath: {best_match['xpath']}")
        else:
            logger.warn(f"Input field with label '{label_text}' not found")
        
        return best_match

    def detect_dropdown_by_text(self, url: str, option_text: str = "", label_text: str = "",
                               confidence_threshold: str = "0.5", scan_full_page: str = "True") -> dict:
        """Detect a dropdown (select) element by option text or label.
        
        Arguments:
        - url: The URL to navigate to
        - option_text: Text of an option in the dropdown (optional)
        - label_text: Label text associated with the dropdown (optional)
        - confidence_threshold: Minimum confidence threshold (default: 0.5)
        - scan_full_page: Whether to scan the full page or just visible area (default: True)
        
        At least one of option_text or label_text must be provided.
        
        Example:
        | ${result}= | Detect Dropdown By Text | https://example.com | United States | Country |
        """
        logger.info(f"Detecting dropdown on {url}")
        confidence_threshold = float(confidence_threshold)
        scan_full_page = scan_full_page.lower() == 'true'
        
        if not option_text and not label_text:
            raise ValueError("Either option_text or label_text must be provided")
        
        self._setup_driver()
        self.driver.get(url)
        time.sleep(2)
        
        if scan_full_page:
            screenshots = self._take_full_page_screenshots()
            all_detections = []
            
            for img, screenshot_path, scroll_offset in screenshots:
                detections = self._run_yolo_detection(img, class_filter=['select', 'dropdown', 'combobox'])
                for detection in detections:
                    detection['bbox'] = self._adjust_bbox_for_scroll(detection['bbox'], scroll_offset)
                    detection['scroll_offset'] = scroll_offset
                    detection['screenshot_path'] = screenshot_path
                all_detections.extend(detections)
        else:
            img, screenshot_path = self._take_screenshot()
            all_detections = self._run_yolo_detection(img, class_filter=['select', 'dropdown', 'combobox'])
            for detection in all_detections:
                detection['scroll_offset'] = 0
                detection['screenshot_path'] = screenshot_path
        
        best_match = None
        best_score = 0
        
        for detection in all_detections:
            if detection['confidence'] < confidence_threshold:
                continue
            
            bbox = detection['bbox']
            scroll_offset = detection.get('scroll_offset', 0)
            
            if scan_full_page:
                elements = self._get_elements_in_bbox_with_scroll(bbox, scroll_offset)
            else:
                elements = self._get_elements_in_bbox(bbox)
            
            for elem_data in elements:
                element = elem_data['element']
                elem_info = self._get_element_info(element)
                
                if elem_info['tagName'].lower() == 'select' or \
                   elem_info.get('role', '').lower() in ['combobox', 'listbox']:
                    
                    max_similarity = 0
                    
                    # Check option text if provided
                    if option_text:
                        try:
                            options = self.driver.execute_script("""
                                var select = arguments[0];
                                var options = [];
                                if (select.tagName.toLowerCase() === 'select') {
                                    for (var i = 0; i < select.options.length; i++) {
                                        options.push(select.options[i].text);
                                    }
                                }
                                return options;
                            """, element)
                            
                            for option in options:
                                similarity = self._text_similarity(option_text, option)
                                if similarity > max_similarity:
                                    max_similarity = similarity
                        except:
                            pass
                    
                    # Check label text if provided
                    if label_text:
                        label_similarity = 0
                        
                        # Method 1: Label with 'for' attribute
                        if elem_info['id']:
                            try:
                                label_element = self.driver.find_element(
                                    By.CSS_SELECTOR, f"label[for='{elem_info['id']}']"
                                )
                                label_info = self._get_element_info(label_element)
                                label_similarity = self._text_similarity(label_text, label_info['text'])
                            except:
                                pass
                        
                        # Method 2: Nearby labels
                        try:
                            nearby_labels = self.driver.execute_script("""
                                var elem = arguments[0];
                                var labels = [];
                                var allLabels = document.querySelectorAll('label');
                                
                                for (var i = 0; i < allLabels.length; i++) {
                                    var labelRect = allLabels[i].getBoundingClientRect();
                                    var elemRect = elem.getBoundingClientRect();
                                    
                                    var distance = Math.abs(labelRect.bottom - elemRect.top) + 
                                                 Math.abs(labelRect.left - elemRect.left);
                                    
                                    if (distance < 100) {
                                        labels.push(allLabels[i]);
                                    }
                                }
                                return labels;
                            """, element)
                            
                            for nearby_label in nearby_labels:
                                label_info = self._get_element_info(nearby_label)
                                similarity = self._text_similarity(label_text, label_info['text'])
                                if similarity > label_similarity:
                                    label_similarity = similarity
                        except:
                            pass
                        
                        if label_similarity > max_similarity:
                            max_similarity = label_similarity
                    
                    # If neither option_text nor label_text provided, just use confidence
                    if not option_text and not label_text:
                        max_similarity = 1.0
                    
                    combined_score = detection['confidence'] * max_similarity
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        xpath = self._get_xpath(element)
                        
                        if scan_full_page:
                            annotated_path = self._create_full_page_annotated_screenshot(
                                screenshots, all_detections, detection
                            )
                        else:
                            annotated_path = self._create_annotated_screenshot(
                                detection['screenshot_path'], all_detections, detection
                            )
                        
                        best_match = {
                            'xpath': xpath,
                            'dom_element': elem_info,
                            'confidence': detection['confidence'],
                            'bbox': bbox,
                            'screenshot_path': detection['screenshot_path'],
                            'annotated_screenshot_path': annotated_path
                        }
        
        if best_match:
            logger.info(f"Dropdown found with confidence {best_match['confidence']:.2f} at xpath: {best_match['xpath']}")
        else:
            logger.warn("Dropdown not found")
        
        return best_match

    def detect_checkbox_radio_by_text(self, url: str, label_text: str, element_type: str = "both",
                                     confidence_threshold: str = "0.5", scan_full_page: str = "True") -> dict:
        """Detect checkbox or radio button by associated label text.
        
        Arguments:
        - url: The URL to navigate to
        - label_text: The label text associated with the checkbox/radio
        - element_type: Type to search for: 'checkbox', 'radio', or 'both' (default: both)
        - confidence_threshold: Minimum confidence threshold (default: 0.5)
        - scan_full_page: Whether to scan the full page or just visible area (default: True)
        
        Example:
        | ${result}= | Detect Checkbox Radio By Text | https://example.com | I agree to terms | checkbox |
        """
        logger.info(f"Detecting {element_type} with label '{label_text}' on {url}")
        confidence_threshold = float(confidence_threshold)
        scan_full_page = scan_full_page.lower() == 'true'
        
        # Set class filter based on element type
        if element_type == 'checkbox':
            class_filter = ['checkbox']
        elif element_type == 'radio':
            class_filter = ['radio']
        else:  # both
            class_filter = ['checkbox', 'radio']
        
        self._setup_driver()
        self.driver.get(url)
        time.sleep(2)
        
        if scan_full_page:
            screenshots = self._take_full_page_screenshots()
            all_detections = []
            
            for img, screenshot_path, scroll_offset in screenshots:
                detections = self._run_yolo_detection(img, class_filter=class_filter)
                for detection in detections:
                    detection['bbox'] = self._adjust_bbox_for_scroll(detection['bbox'], scroll_offset)
                    detection['scroll_offset'] = scroll_offset
                    detection['screenshot_path'] = screenshot_path
                all_detections.extend(detections)
        else:
            img, screenshot_path = self._take_screenshot()
            all_detections = self._run_yolo_detection(img, class_filter=class_filter)
            for detection in all_detections:
                detection['scroll_offset'] = 0
                detection['screenshot_path'] = screenshot_path
        
        best_match = None
        best_score = 0
        
        for detection in all_detections:
            if detection['confidence'] < confidence_threshold:
                continue
            
            bbox = detection['bbox']
            scroll_offset = detection.get('scroll_offset', 0)
            
            if scan_full_page:
                elements = self._get_elements_in_bbox_with_scroll(bbox, scroll_offset)
            else:
                elements = self._get_elements_in_bbox(bbox)
            
            for elem_data in elements:
                element = elem_data['element']
                elem_info = self._get_element_info(element)
                
                if elem_info['tagName'].lower() == 'input' and \
                   elem_info['type'].lower() in ['checkbox', 'radio']:
                    
                    max_similarity = 0
                    
                    # Method 1: Label with 'for' attribute
                    if elem_info['id']:
                        try:
                            label_element = self.driver.find_element(
                                By.CSS_SELECTOR, f"label[for='{elem_info['id']}']"
                            )
                            label_info = self._get_element_info(label_element)
                            similarity = self._text_similarity(label_text, label_info['text'])
                            if similarity > max_similarity:
                                max_similarity = similarity
                        except:
                            pass
                    
                    # Method 2: Parent label
                    try:
                        parent_label = self.driver.execute_script("""
                            var elem = arguments[0];
                            var parent = elem.parentElement;
                            while (parent && parent.tagName.toLowerCase() !== 'label') {
                                parent = parent.parentElement;
                            }
                            return parent;
                        """, element)
                        
                        if parent_label:
                            label_info = self._get_element_info(parent_label)
                            similarity = self._text_similarity(label_text, label_info['text'])
                            if similarity > max_similarity:
                                max_similarity = similarity
                    except:
                        pass
                    
                    # Method 3: Adjacent label (next sibling)
                    try:
                        adjacent_label = self.driver.execute_script("""
                            var elem = arguments[0];
                            var next = elem.nextElementSibling;
                            if (next && next.tagName.toLowerCase() === 'label') {
                                return next;
                            }
                            var prev = elem.previousElementSibling;
                            if (prev && prev.tagName.toLowerCase() === 'label') {
                                return prev;
                            }
                            return null;
                        """, element)
                        
                        if adjacent_label:
                            label_info = self._get_element_info(adjacent_label)
                            similarity = self._text_similarity(label_text, label_info['text'])
                            if similarity > max_similarity:
                                max_similarity = similarity
                    except:
                        pass
                    
                    # Method 4: Nearby text elements
                    try:
                        nearby_text = self.driver.execute_script("""
                            var elem = arguments[0];
                            var elemRect = elem.getBoundingClientRect();
                            var allElements = document.querySelectorAll('*');
                            var nearbyTexts = [];
                            
                            for (var i = 0; i < allElements.length; i++) {
                                var el = allElements[i];
                                var rect = el.getBoundingClientRect();
                                var text = el.textContent || el.innerText || '';
                                
                                if (text.trim() && 
                                    Math.abs(rect.left - elemRect.right) < 200 && 
                                    Math.abs(rect.top - elemRect.top) < 50) {
                                    nearbyTexts.push(text.trim());
                                }
                            }
                            return nearbyTexts;
                        """, element)
                        
                        for text in nearby_text:
                            similarity = self._text_similarity(label_text, text)
                            if similarity > max_similarity:
                                max_similarity = similarity
                    except:
                        pass
                    
                    combined_score = detection['confidence'] * max_similarity
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        xpath = self._get_xpath(element)
                        
                        if scan_full_page:
                            annotated_path = self._create_full_page_annotated_screenshot(
                                screenshots, all_detections, detection
                            )
                        else:
                            annotated_path = self._create_annotated_screenshot(
                                detection['screenshot_path'], all_detections, detection
                            )
                        
                        best_match = {
                            'xpath': xpath,
                            'dom_element': elem_info,
                            'confidence': detection['confidence'],
                            'bbox': bbox,
                            'screenshot_path': detection['screenshot_path'],
                            'annotated_screenshot_path': annotated_path
                        }
        
        if best_match:
            logger.info(f"{element_type.title()} found with confidence {best_match['confidence']:.2f} at xpath: {best_match['xpath']}")
        else:
            logger.warn(f"{element_type.title()} with label '{label_text}' not found")
        
        return best_match

    def detect_element_by_proximity(self, url: str, reference_text: str, target_element_type: str,
                                   max_distance: str = "200", direction: str = "any",
                                   confidence_threshold: str = "0.5", scan_full_page: str = "True") -> dict:
        """Detect an element by its proximity to a reference text or label.
        
        Arguments:
        - url: The URL to navigate to
        - reference_text: Text to use as reference point
        - target_element_type: Type of element to find ('input', 'button', 'select', 'checkbox', 'radio')
        - max_distance: Maximum distance in pixels from reference text (default: 200)
        - direction: Direction to search ('any', 'right', 'left', 'below', 'above') (default: any)
        - confidence_threshold: Minimum confidence threshold (default: 0.5)
        - scan_full_page: Whether to scan the full page or just visible area (default: True)
        
        Example:
        | ${result}= | Detect Element By Proximity | https://example.com | Password | input | 150 | right |
        """
        logger.info(f"Detecting {target_element_type} near text '{reference_text}' on {url}")
        confidence_threshold = float(confidence_threshold)
        max_distance = int(max_distance)
        scan_full_page = scan_full_page.lower() == 'true'
        
        # Set class filter based on target element type
        if target_element_type == 'input':
            class_filter = ['input', 'textbox']
        elif target_element_type == 'button':
            class_filter = ['button']
        elif target_element_type == 'select':
            class_filter = ['select', 'dropdown', 'combobox']
        elif target_element_type == 'checkbox':
            class_filter = ['checkbox']
        elif target_element_type == 'radio':
            class_filter = ['radio']
        else:
            class_filter = None  # Search all types
        
        self._setup_driver()
        self.driver.get(url)
        time.sleep(2)
        
        if scan_full_page:
            screenshots = self._take_full_page_screenshots()
            all_detections = []
            
            for img, screenshot_path, scroll_offset in screenshots:
                detections = self._run_yolo_detection(img, class_filter=class_filter)
                for detection in detections:
                    detection['bbox'] = self._adjust_bbox_for_scroll(detection['bbox'], scroll_offset)
                    detection['scroll_offset'] = scroll_offset
                    detection['screenshot_path'] = screenshot_path
                all_detections.extend(detections)
        else:
            img, screenshot_path = self._take_screenshot()
            all_detections = self._run_yolo_detection(img, class_filter=class_filter)
            for detection in all_detections:
                detection['scroll_offset'] = 0
                detection['screenshot_path'] = screenshot_path
        
        # Find reference elements with matching text
        reference_elements = self.driver.execute_script("""
            var searchText = arguments[0];
            var allElements = document.querySelectorAll('*');
            var references = [];
            
            for (var i = 0; i < allElements.length; i++) {
                var el = allElements[i];
                var text = el.textContent || el.innerText || '';
                
                if (text.trim() && text.toLowerCase().includes(searchText.toLowerCase())) {
                    var rect = el.getBoundingClientRect();
                    var scrollY = window.pageYOffset || document.documentElement.scrollTop;
                    var scrollX = window.pageXOffset || document.documentElement.scrollLeft;
                    
                    references.push({
                        element: el,
                        text: text.trim(),
                        rect: {
                            left: rect.left + scrollX,
                            top: rect.top + scrollY,
                            right: rect.right + scrollX,
                            bottom: rect.bottom + scrollY
                        }
                    });
                }
            }
            return references;
        """, reference_text)
        
        if not reference_elements:
            logger.warn(f"Reference text '{reference_text}' not found")
            return None
        
        best_match = None
        best_score = 0
        
        for detection in all_detections:
            if detection['confidence'] < confidence_threshold:
                continue
            
            bbox = detection['bbox']
            scroll_offset = detection.get('scroll_offset', 0)
            
            if scan_full_page:
                elements = self._get_elements_in_bbox_with_scroll(bbox, scroll_offset)
            else:
                elements = self._get_elements_in_bbox(bbox)
            
            for elem_data in elements:
                element = elem_data['element']
                elem_info = self._get_element_info(element)
                
                # Convert element rect to absolute coordinates
                element_rect = elem_info['rect']
                current_scroll = self.driver.execute_script("""
                    return {
                        x: window.pageXOffset || document.documentElement.scrollLeft, 
                        y: window.pageYOffset || document.documentElement.scrollTop
                    };
                """)
                element_rect_abs = {
                    'left': element_rect['left'] + current_scroll['x'],
                    'top': element_rect['top'] + current_scroll['y'],
                    'right': element_rect['right'] + current_scroll['x'],
                    'bottom': element_rect['bottom'] + current_scroll['y']
                }
                
                # Check proximity to each reference element
                for ref_data in reference_elements:
                    ref_rect = ref_data['rect']
                    
                    # Calculate distance
                    distance = self._calculate_distance(element_rect_abs, ref_rect)
                    
                    if distance > max_distance:
                        continue
                    
                    # Check direction if specified
                    if direction != 'any':
                        element_center_x = (element_rect_abs['left'] + element_rect_abs['right']) / 2
                        element_center_y = (element_rect_abs['top'] + element_rect_abs['bottom']) / 2
                        ref_center_x = (ref_rect['left'] + ref_rect['right']) / 2
                        ref_center_y = (ref_rect['top'] + ref_rect['bottom']) / 2
                        
                        if direction == 'right' and element_center_x <= ref_center_x:
                            continue
                        elif direction == 'left' and element_center_x >= ref_center_x:
                            continue
                        elif direction == 'below' and element_center_y <= ref_center_y:
                            continue
                        elif direction == 'above' and element_center_y >= ref_center_y:
                            continue
                    
                    # Calculate text similarity for better matching
                    text_similarity = self._text_similarity(reference_text, ref_data['text'])
                    
                    # Calculate combined score (closer is better, higher confidence is better)
                    distance_score = max(0, (max_distance - distance) / max_distance)
                    combined_score = detection['confidence'] * text_similarity * distance_score
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        xpath = self._get_xpath(element)
                        
                        if scan_full_page:
                            annotated_path = self._create_full_page_annotated_screenshot(
                                screenshots, all_detections, detection
                            )
                        else:
                            annotated_path = self._create_annotated_screenshot(
                                detection['screenshot_path'], all_detections, detection
                            )
                        
                        best_match = {
                            'xpath': xpath,
                            'dom_element': elem_info,
                            'confidence': detection['confidence'],
                            'bbox': bbox,
                            'screenshot_path': detection['screenshot_path'],
                            'annotated_screenshot_path': annotated_path
                        }
        
        if best_match:
            logger.info(f"Element found with confidence {best_match['confidence']:.2f} at xpath: {best_match['xpath']}")
        else:
            logger.warn(f"No {target_element_type} found near '{reference_text}'")
        
        return best_match

    def detect_icon_by_image(self, url: str, icon_image_path: str, 
                           confidence_threshold: str = "0.5", scan_full_page: str = "True") -> dict:
        """Detect an icon by comparing with a reference image.
        
        Arguments:
        - url: The URL to navigate to
        - icon_image_path: Path to the reference icon image
        - confidence_threshold: Minimum confidence threshold (default: 0.5)
        - scan_full_page: Whether to scan the full page or just visible area (default: True)
        
        Example:
        | ${result}= | Detect Icon By Image | https://example.com | /path/to/search_icon.png |
        """
        logger.info(f"Detecting icon using reference image '{icon_image_path}' on {url}")
        confidence_threshold = float(confidence_threshold)
        scan_full_page = scan_full_page.lower() == 'true'
        
        self._setup_driver()
        self.driver.get(url)
        time.sleep(2)
        
        ref_icon = cv2.imread(icon_image_path)
        if ref_icon is None:
            raise ValueError(f"Could not load reference icon: {icon_image_path}")
        
        if scan_full_page:
            screenshots = self._take_full_page_screenshots()
            all_detections = []
            
            for img, screenshot_path, scroll_offset in screenshots:
                detections = self._run_yolo_detection(img, class_filter=['icon', 'image'])
                for detection in detections:
                    detection['bbox'] = self._adjust_bbox_for_scroll(detection['bbox'], scroll_offset)
                    detection['scroll_offset'] = scroll_offset
                    detection['screenshot_path'] = screenshot_path
                    detection['img'] = img
                all_detections.extend(detections)
        else:
            img, screenshot_path = self._take_screenshot()
            all_detections = self._run_yolo_detection(img, class_filter=['icon', 'image'])
            for detection in all_detections:
                detection['scroll_offset'] = 0
                detection['screenshot_path'] = screenshot_path
                detection['img'] = img
        
        best_match = None
        best_score = 0
        
        for detection in all_detections:
            if detection['confidence'] < confidence_threshold:
                continue
            
            bbox = detection['bbox']
            scroll_offset = detection.get('scroll_offset', 0)
            
            if scan_full_page:
                orig_bbox = (bbox[0], bbox[1] - scroll_offset, bbox[2], bbox[3] - scroll_offset)
            else:
                orig_bbox = bbox
                
            x1, y1, x2, y2 = orig_bbox
            
            img_cv = cv2.cvtColor(detection['img'], cv2.COLOR_RGB2BGR)
            detected_region = img_cv[y1:y2, x1:x2]
            
            if detected_region.size == 0:
                continue
            
            h, w = detected_region.shape[:2]
            ref_resized = cv2.resize(ref_icon, (w, h))
            
            result = cv2.matchTemplate(detected_region, ref_resized, cv2.TM_CCOEFF_NORMED)
            similarity = np.max(result)
            combined_score = detection['confidence'] * similarity
            
            if combined_score > best_score:
                best_score = combined_score
                
                if scan_full_page:
                    elements = self._get_elements_in_bbox_with_scroll(bbox, scroll_offset)
                else:
                    elements = self._get_elements_in_bbox(bbox)
                    
                if elements:
                    element = elements[0]['element']
                    xpath = self._get_xpath(element)
                    elem_info = self._get_element_info(element)
                    
                    if scan_full_page:
                        annotated_path = self._create_full_page_annotated_screenshot(
                            screenshots, all_detections, detection
                        )
                    else:
                        annotated_path = self._create_annotated_screenshot(
                            detection['screenshot_path'], all_detections, detection
                        )
                    
                    best_match = {
                        'xpath': xpath,
                        'dom_element': elem_info,
                        'confidence': detection['confidence'],
                        'bbox': bbox,
                        'screenshot_path': detection['screenshot_path'],
                        'annotated_screenshot_path': annotated_path
                    }
        
        if best_match:
            logger.info(f"Icon found with confidence {best_match['confidence']:.2f} at xpath: {best_match['xpath']}")
        else:
            logger.warn("Icon not found")
        
        return best_match
    
    def get_element_xpath(self, result: dict) -> str:
        """Get the XPath from a detection result.
        
        Arguments:
        - result: Detection result dictionary from any detect keyword
        
        Returns the XPath string.
        
        Example:
        | ${result}= | Detect Button By Text | https://example.com | Submit |
        | ${xpath}= | Get Element XPath | ${result} |
        """
        if result and 'xpath' in result:
            return result['xpath']
        return ""
    
    def get_element_info(self, result: dict) -> dict:
        """Get detailed DOM element information from a detection result.
        
        Arguments:
        - result: Detection result dictionary from any detect keyword
        
        Returns element information dictionary.
        
        Example:
        | ${result}= | Detect Button By Text | https://example.com | Submit |
        | ${info}= | Get Element Info | ${result} |
        | Log | Element tag: ${info.tagName} |
        """
        if result and 'dom_element' in result:
            return result['dom_element']
        return {}
    
    def get_detection_confidence(self, result: dict) -> float:
        """Get the confidence score from a detection result.
        
        Arguments:
        - result: Detection result dictionary from any detect keyword
        
        Returns the confidence score as a float.
        
        Example:
        | ${result}= | Detect Button By Text | https://example.com | Submit |
        | ${confidence}= | Get Detection Confidence | ${result} |
        | Should Be True | ${confidence} > 0.5 |
        """
        if result and 'confidence' in result:
            return float(result['confidence'])
        return 0.0
    
    def close_browser(self):
        """Close the browser and cleanup resources.
        
        Example:
        | Close Browser |
        """
        logger.info("Closing browser")
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.close_browser()