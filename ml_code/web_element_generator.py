import os as os
import traceback
from io import BytesIO

import tensorflow as tf
from PIL import Image

import app_facing_code as appcode
import string_int_label_map_pb2 
import util_classifier as ut
from scipy.misc import imread, imsave, imresize


tf.enable_eager_execution()
script_dir = os.path.dirname(__file__)
dir_path = '/Users/bardek01/Personal/projects/new_try_tensorflow/models/research/models/model_2/'
abs_image_path = os.path.join(script_dir, dir_path)
util_obj = ut.Utilities()
labelmap_file = dir_path + 'label_map.pbtxt'


class WebElementGenerator:

    def __init__(self, driver):
        self.driver = driver

    def get_suffix_from_element(self,element):
        
        if element.text:
            return element.text.replace(" ","_")  
        elif element.get_attribute("title"):
             return element.get_attribute("title").replace(" ","_")
        elif element.get_attribute("value"):
            element.get_attribute("value").replace(" ","_")

    def generate_elements(self, url):
        train_writer = tf.python_io.TFRecordWriter(dir_path + 'train.record')
        test_writer = tf.python_io.TFRecordWriter(dir_path + 'test.record')

        try:
            if os.path.isfile(labelmap_file) == True:
                label_file = util_obj.open_file(labelmap_file, "rb")
                class_map = string_int_label_map_pb2.StringIntLabelMap()
                class_map.ParseFromString(label_file.read())
                label_file.close()
            else:
                class_map = string_int_label_map_pb2.StringIntLabelMap()

            self.driver.get(url)
            
            all_visible_elements = self.driver.find_elements_by_xpath("//a[not(contains(@style,'display:none'))] | //h2[not(contains(@style,'display:none'))] | //button[not(contains(@style,'display:none'))] | //span[not(contains(@style,'display:none'))] | //input[not(contains(@style,'display:none'))]")

            elements_list = []

            for element in all_visible_elements:
                
                location = element.location_once_scrolled_into_view
                size = element.size
                x1 = location['x'] * 2
                y1 = location['y'] * 2
                x2 = x1 + size['width'] * 2
                y2 = y1 + size['height'] * 2
                words = element.text.split()
                if x2 != 0 and y2 != 0 and len(words) < 4:
                    
                    png = self.driver.get_screenshot_as_png()
                    img = Image.open(BytesIO(png))
                    img = img.crop((x1, y1, x2, y2))
                    suffix = self.get_suffix_from_element(element)

                    if not suffix:
                        continue

                    class_name = element.tag_name + '_' + suffix
                    original_image_path = dir_path + "images/train/" + class_name + '.png'
                    img.save(original_image_path)
                    

                    # writing class to label_map.pbtxt
                    class_id = util_obj.add_class_to_label_map(class_name, class_map)
                    print("class id", class_id)
                    # creating an tf.example
                    width = x2 - x1
                    height = y2 - x1
                    #example = util_obj.create_tf_example_object_detection(width, height, original_image_path, img, class_name, class_id)
                    #writer.write(example.SerializeToString())

                    elements_list.append([original_image_path,width,height,class_name,class_id])
        except Exception:
                print(class_name)
                traceback.print_exc()

        train_list,test_list = util_obj.create_augmented_images(elements_list,50)

        self.serialize_image_list(train_writer,train_list)
        self.serialize_image_list(test_writer,test_list)


        label_map_file = util_obj.open_file(labelmap_file, "wb")
        label_map_file.write(class_map.SerializeToString())
        label_map_file.close()

        train_writer.close()
        test_writer.close()

    def serialize_image_list(self,tf_writer,images_list):
        
        for image_list in images_list:
            example = util_obj.create_tf_example_object_detection(image_list[1], image_list[2], image_list[0], Image.open(image_list[0]), image_list[3], image_list[4])
            tf_writer.write(example.SerializeToString())

if __name__ == '__main__':
    driver = appcode.AppFacing('pc', False).get_driver()
    web_element_generator = WebElementGenerator(driver)
    web_element_generator.generate_elements("https://www.bbc.co.uk/")
    driver.quit()
