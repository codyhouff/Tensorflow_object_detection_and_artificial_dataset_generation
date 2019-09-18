from PIL import Image, ImageEnhance, ImageFilter
import random 
import glob
import os
import csv


# determine the directory for the front image, background image, and the folder where the images will be saved
################################################################################
                                                                    
background_directory = "backgrounds/(121)_small/"  # path to backgrounds

images_directory = "images/"                       # output path to the test and train image folder

front_directory = "front/street_signs/"            # path to the front object images 

###############################################################################


def find_num_train_test_imgs(background_directory):
    num_back_imgs = len(os.listdir(background_directory)) # find the number of background images
    num_train_imgs = int(num_back_imgs * .90)       # determine the number of train images
    num_test_imgs = num_back_imgs - num_train_imgs  # determine the number of test images

    return num_train_imgs, num_test_imgs


def initialize_train_and_test_csv():
    csv_save_name = 'data/train_labels.csv'
    f = open(csv_save_name, 'w', newline ='')
    thewriter = csv.writer(f)
    thewriter.writerow(['filename','width','height','class','xmin','ymin','xmax','ymax'])
    f.close()

    csv_save_name = 'data/test_labels.csv'
    f = open(csv_save_name, 'w', newline ='')
    thewriter = csv.writer(f)
    thewriter.writerow(['filename','width','height','class','xmin','ymin','xmax','ymax'])
    f.close()


def get_class_name(front_filename):
    class_name = front_filename
    class_name = class_name.replace('.png','')
    class_name = class_name.replace('front/','')
    class_name = class_name.replace('street_signs\\','')
    
    return class_name


def open_front_and_background_imgs(front_filename, background_filename):
    img = Image.open(front_filename)
    front_list.append(img)
    img_w, img_h = img.size

    background = Image.open(background_filename)
    image_list.append(background)
    bg_w, bg_h = background.size

    return img, img_w, img_h, background, bg_w, bg_h


def change_front_and_back_characteristics(img, img_w, img_h, background, bg_w, bg_h):
    img = color_img(img) # color the img
    background = color_img(background) # color the background
    
    img = contrast_img(img) # contrast the img
    background = contrast_img(background) # contrast the background
        
    img = change_brightness(img) # change img brightness
    background = change_brightness(background) # change background brightness

    img, img_w, img_h = stretch_img(img_w, img_h, img) #stretch

    img = rotate_img(img) # change rotation 
       
    img, img_w, img_h = resize_img(bg_w, bg_h, img_w, img_h, img) # resize

    new_location = relocate_img(bg_w, bg_h, img_w, img_h) # relocate
    new_w, new_h = new_location

    x_max = new_w + img_w
    y_max = new_h + img_h

    return img, bg_w, bg_h, new_w, new_h, x_max, y_max, new_location


def create_and_save_new_img(img, background, save_path, new_location, iteration_1, iteration_2):
    background.paste(img, new_location, img)

    save_name = "train_image_" + str(iteration_1) + "." + str(iteration_2) + ".jpg"    
    full_save_path = save_path + save_name
    background.save(full_save_path)

    return save_name
 

def stretch_img(img_w, img_h, img):
    random_num = random.randrange(1, 101, 1)
    random_stretch = random.randrange(60, 100, 1)/100

    if(random_num <= 20): # 20% chance width changes # if the smallest side on the background image is the width
        size = (int(img_w * random_stretch), img_h)
        img = img.resize(size,Image.ANTIALIAS)
        img_w, img_h = img.size

    elif((random_num >= 20)&(random_num <= 40)):  #20% chance height changes # if the smallest side on the background image is the height or both sides are equal
        size = (img_w, int(img_h * random_stretch))
        img = img.resize(size,Image.ANTIALIAS)
        img_w, img_h = img.size
                                   #60% chance nothing changes
    return img, img_w, img_h


def color_img(img):
    enhancer = ImageEnhance.Color(img)
    new_color = random.randrange(10, 200, 1)/100
    colored_img = enhancer.enhance(new_color)
    return colored_img


def contrast_img(img):
    enhancer = ImageEnhance.Contrast(img) 
    new_contrast = random.randrange(10, 200, 1)/100
    contrasted_img = enhancer.enhance(new_contrast)
    return contrasted_img 


def change_brightness(img):
    enhancer = ImageEnhance.Brightness(img)
    new_brightness = random.randrange(10, 130, 1)/100
    brightened_img = enhancer.enhance(new_brightness)
    return brightened_img


def rotate_img(img):
    new_rotation = random.randrange(-15, 15, 1)
    img = img.rotate(new_rotation)
    return img


def resize_img(bg_w, bg_h, img_w, img_h, img):
    change_size = random.randrange(6, 30, 1)/100

    if(bg_w < bg_h):    # if the smallest side on the background image is the width
        new_w = int(bg_w * change_size)
        new_h = int(img_h * (new_w / img_w))
        size = (new_w, new_h)
    else:               # if the smallest side on the background image is the height or both sides ar equal
        new_h = int(bg_h * change_size)
        new_w = int(img_w * (new_h / img_h))
        size = (new_w, new_h)
        
    img = img.resize(size,Image.ANTIALIAS)
    img_w, img_h = img.size

    return img, img_w, img_h


def relocate_img(bg_w, bg_h, img_w, img_h):
    mid_w = (bg_w - img_w) // 2
    mid_h = (bg_h - img_h) // 2
    move_w = random.randrange(5, 195, 1)/100
    move_h = random.randrange(5, 195, 1)/100
    offset = (int(mid_w * move_w), int(mid_h * move_h))
    return offset


num_train_imgs, num_test_imgs = find_num_train_test_imgs(background_directory) # determine the number of train and test imgs based on the number of background imgs

initialize_train_and_test_csv() # initialize the train and test csv with ['filename','width','height','class','xmin','ymin','xmax','ymax']

iteration_1 = 1
front_list = []
for front_filename in glob.glob(front_directory + "*.png"): # go through each image in the front folder


    class_name = get_class_name(front_filename)

    iteration_2 = 1
    image_list = []
    for background_filename in glob.glob(background_directory + "*.jpg"): # go through each image in the background folder

        if(iteration_2 <= num_train_imgs):    # split the training and test data 90% train and 10% test
            csv_save_name = 'data/train_labels.csv'
            with open(csv_save_name, 'a', newline ='') as tr:
                thewriter = csv.writer(tr)                   
                save_path = images_directory + "train/"

                img, img_w, img_h, background, bg_w, bg_h = open_front_and_background_imgs(front_filename, background_filename) # open front and background img and find width and height of both

                img, bg_w, bg_h, new_w, new_h, x_max, y_max, new_location = change_front_and_back_characteristics(img, img_w, img_h, background, bg_w, bg_h) # change change front and background characteristics

                save_name = create_and_save_new_img(img, background, save_path, new_location, iteration_1, iteration_2) # create a new image with the front image placed on the background image and save the new img
        
                thewriter.writerow([save_name,str(bg_w),str(bg_h),class_name,str(new_w),str(new_h),str(x_max),str(y_max)]) # add a new row to the csv file with the locations of the bounding box and image size
                
        elif(iteration_2 > num_train_imgs):  # split the training and test data 90% train and 10% test
            csv_save_name = 'data/test_labels.csv'
            with open(csv_save_name, 'a', newline ='') as te:
                thewriter = csv.writer(te)
                save_path = images_directory + "test/"

                img, img_w, img_h, background, bg_w, bg_h = open_front_and_background_imgs(front_filename, background_filename) # open front and background img and find width and height of both

                img, bg_w, bg_h, new_w, new_h, x_max, y_max, new_location = change_front_and_back_characteristics(img, img_w, img_h, background, bg_w, bg_h) # change change front and background characteristics

                save_name = create_and_save_new_img(img, background, save_path, new_location, iteration_1, iteration_2) # create a new image with the front image placed on the background image and save the new img
        
                thewriter.writerow([save_name,str(bg_w),str(bg_h),class_name,str(new_w),str(new_h),str(x_max),str(y_max)]) # add a new row to the csv file with the locations of the bounding box and image size
                

        print("iteration: " + str(iteration_1) + "." + str(iteration_2))
        
        iteration_2 = iteration_2 + 1 # background image number
        
    tr.close()
    te.close()
    iteration_1 = iteration_1 + 1 # front image number
