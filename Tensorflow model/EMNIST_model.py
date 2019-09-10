from progressBar import progressBar
import tensorflow as tf
import numpy as np 
import sys
import math
import argparse
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

model_name = "model3"

itr = 1
epoch_count = 500
batch_size = 1
label_count = 47
learning_rate = 0.001
image_size = 28 * 28
image_shape = [-1,28,28,1]
label_shape = [-1,label_count]
progress = progressBar(tot_items = 30)

#session
sess = tf.Session();

#tensorboard
writer = tf.summary.FileWriter("./Logs/NN_log",sess.graph);
merged = tf.summary.merge_all();


Input_image = tf.placeholder(tf.float64 );
Input_image_reshaped = tf.reshape(Input_image,image_shape);

label_matrix = tf.placeholder(tf.float64);
label_matrix_reshaped = tf.reshape(label_matrix,label_shape);

#model : 
conv1 = tf.layers.conv2d(inputs = Input_image_reshaped, filters = 20, kernel_size = [3,3], padding='SAME', activation = tf.nn.relu) #input : 28 x 28 x 1
pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = 2, strides = 2)     
conv2 = tf.layers.conv2d(inputs = pool1, filters = 40, kernel_size = [3,3], padding='SAME', activation = tf.nn.relu)    #input : 14 x 14 x 64
pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = 2, strides = 2)
conv3 = tf.layers.conv2d(inputs = pool2, filters = 200, kernel_size = [5,5], padding='VALID', activation = tf.nn.relu)    #input : 5 x 5 x 128

dense0 = tf.reshape(conv3, [-1,1800])
dense1 = tf.layers.dense(inputs = dense0, units = 300, activation = tf.nn.relu)
dense2 = tf.layers.dense(inputs = dense1, units = label_count)

classifier = tf.nn.softmax(dense2)
Error = tf.losses.softmax_cross_entropy(label_matrix_reshaped, dense2);
Optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(Error);
tf.summary.scalar("Loss function",Error);

#weight logger
Saver = tf.train.Saver();

mapping = []
image_input_list = []


def get_data_and_labels(images_filename, labels_filename, file_type):
    print("Opening " + file_type + " files ...")
    images_file = open(images_filename, "rb")
    labels_file = open(labels_filename, "rb")

    try:
        print("Reading "+ file_type +" files ...")
        images_file.read(4)
        num_of_items = int.from_bytes(images_file.read(4), byteorder="big")
        num_of_rows = int.from_bytes(images_file.read(4), byteorder="big")
        num_of_colums = int.from_bytes(images_file.read(4), byteorder="big")
        labels_file.read(8)

        num_of_image_values = num_of_rows * num_of_colums
        data = [[None for x in range(num_of_image_values)]
                for y in range(num_of_items)]
        labels = []
        progress.reset(num_of_items)
        for item in range(num_of_items):
            for value in range(num_of_image_values):
                data[item][value] = int.from_bytes(images_file.read(1),
                                                   byteorder="big")
            for i in range(28):
                for j in range(i):
                    data[item][i*28 + j] , data[item][j*28 + i] = data[item][j*28 + i] , data[item][i*28 + j]

            labels.append(int.from_bytes(labels_file.read(1), byteorder="big"))
            progress.signal_job_done()
        D = []
        for image in data:
            for pixel in image:
                D.append(pixel)
        return D, labels
    finally:
        images_file.close()
        labels_file.close()
        print("Files closed.")


def initialize(InitMode,FileAdd = ""):
    if InitMode == "DEFAULT":
        sess.run(tf.global_variables_initializer());
        print("Network initialized with random initializer");
    if InitMode == "LOAD":
        Saver.restore(sess, FileAdd);
        print("Network initialized with requested values");


def one_hot(labels):
    oneHot = np.zeros(label_count * len(labels));

    for i in range(len(labels)):
        label = labels[i]
        oneHot[label_count * i + label] = 1.0;
    return oneHot


def log_index(index,labels, data):
    fist = data[index];
    for i in range(28):
        Str = '';
        for j in range(28):
            if fist[i*28 + j] > 127:
                Str += '@';
            else:
                Str += ' ';
        print(Str);


def _test(test_data, test_label):
    classify = sess.run(classifier, feed_dict = {Input_image:[test_data]})
    classes = []
    for Class in classify:
        classes.append(np.argmax(Class))
    #print(classes)
    cnt = 0;
    for i in range(len(test_label)):
        if test_label[i] == classes[i]:
            cnt += 1;

    return float(cnt * 100) / float(len(test_label))


def train_a_batch(ind, train_data, train_label_one_hot):
    images = train_data[ind * image_size * batch_size:min((ind+1) * image_size * batch_size, len(train_data))]
    labels = train_label_one_hot[ind * label_count * batch_size:min((ind+1) * label_count * batch_size, len(train_label_one_hot))]
    sess.run(Optimizer , feed_dict={ Input_image : [images] , label_matrix: [labels] })


def Train(train_data_dir, test_data_dir):
    train_data , train_label = get_data_and_labels(train_data_dir + 'emnist-balanced-train-images-idx3-ubyte' ,\
                                                    train_data_dir + 'emnist-balanced-train-labels-idx1-ubyte' , 'train')

    test_data , test_label = get_data_and_labels(test_data_dir + 'emnist-balanced-test-images-idx3-ubyte' ,\
                                                    test_data_dir + 'emnist-balanced-test-labels-idx1-ubyte' , 'test')

    train_label_one_hot = one_hot(train_label) 
    print("One hot conversion completed.")

    for i in range(epoch_count):
        batch_count = int(len(train_label) / batch_size)
        if len(train_label) % batch_size > 0:
            batch_count += 1;
        rand_index = random.randint(0,len(train_label))
        loss = sess.run(Error , feed_dict={ Input_image : [train_data[rand_index * image_size:(rand_index + 1)*image_size]] , label_matrix: [train_label_one_hot[(rand_index) * label_count:(rand_index +1) * label_count]] })
        #accuracy = _test(test_data, test_label)
        #print("Epoch : " + str(i + 1) + "/" + str(epoch_count) + '\t' + "Loss : " + str(loss) + '\t' + "Accuracy : " + str(accuracy) + "%" )
        print("Epoch : " + str(i + 1) + "/" + str(epoch_count) + '\t' + "Loss : " + str(loss))
        progress.reset(batch_count);

        for j in range(batch_count):
            for t in range(itr):
                train_a_batch(j, train_data,train_label_one_hot)
            progress.signal_job_done()

        Saver.save(sess,"./"+model_name+"/Weights");    
        

def Test(test_data_dir):
    test_data , test_label = get_data_and_labels(test_data_dir + 'emnist-balanced-test-images-idx3-ubyte' ,\
                                                    test_data_dir + 'emnist-balanced-test-labels-idx1-ubyte' , 'test')
    cnt = 0;
    progress.reset(len(test_label));
    for ind in range(len(test_label)):
        classify = sess.run(classifier, feed_dict = {Input_image:[test_data[ind * image_size:(ind + 1)*image_size]]})
        if test_label[ind] == np.argmax(classify):
            cnt += 1;
        progress.signal_job_done()
    return float(cnt * 100) / float(len(test_label))


def get_mapping(fileAddr="../Balanced/emnist-balanced-mapping.txt"):
    print("Loading Mapping...")
    mapping = {};
    mappings = open(fileAddr,"r").readlines();
    for Map in mappings:
        L = Map.split(' ');
        mapping[int(L[0])] = int(L[1])
    print("Mapping is loaded.");
    return mapping


def convert_to_black_and_white(img):

    IMG = np.zeros((img.shape[0],img.shape[1]), dtype=float)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            IMG[i][j] = np.average(img[i][j][0:3])
    return IMG


def REVERSE(img):
    IMG = np.zeros((img.shape[0],img.shape[1]))
    IMG2 = np.zeros((img.shape[0],img.shape[1]))
    for i in range(len(img)):
        for j in range(len(img[i])):
            IMG[i][j] = 255.0 - float(img[i][j])

    return IMG


def load_image_input(address):
    global image_input_list
    image_input_list = os.listdir(address) 
    image_inputs = []

    print(image_input_list)

    for addr in image_input_list:
        img = Image.open(address + addr)
        img = np.array(img)
        if len(img.shape) == 3:
            img = convert_to_black_and_white(img)
        img = REVERSE(img)
        image_inputs.append(img)

    return image_inputs
   

def repair(img, img_cpy):
    tmp = np.zeros( (img.shape[0],img.shape[1]) )
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > 0:
                tmp[i][j] = img_cpy[i][j]
    return tmp


def vertical_abstract(img):
    res = []
    for i in range(img.shape[0]):
        l = []
        stat = False
        for j in range(img.shape[1]):
            if stat and img[i][j] == 0:
                l = np.array(l)
                res.append(np.mean(l,axis = 0))
                stat = False
                l = []

            if stat == False and img[i][j] > 0:
                l.append([i,j])
                stat = True

            if stat and img[i][j] > 0:
                l.append([i,j])

            if stat == False and img[i][j] == 0:
                continue;
    return res;


def horizontal_abstract(img):
    res = []
    for i in range(img.shape[1]):
        l = []
        stat = False
        for j in range(img.shape[0]):
            if stat and img[j][i] == 0:
                l = np.array(l)
                res.append(np.mean(l,axis = 0))
                stat = False
                l = []

            if stat == False and img[j][i] > 0:
                l.append([j,i])
                stat = True

            if stat and img[j][i] > 0:
                l.append([j,i])

            if stat == False and img[j][i] == 0:
                continue;
    return res;


def relax(img,window):
    tmp = np.zeros( (img.shape[0],img.shape[1]) )
    for i in range(0,img.shape[0],window):
        for j in range(0,img.shape[1],window):
            if i < img.shape[0] - window - 1 and j < img.shape[1] - window - 1:
                x = np.mean(img[i:i+window,j:j+window])
                tmp[i:i+window,j:j+window] = x
    img= tmp
    return img


def filter(img,thresh = 127):
    tmp = np.zeros( (img.shape[0],img.shape[1]) )
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > thresh:
                tmp[i][j] = 255
    img= tmp
    return img
    

def dist(pix1,pix2):
    return np.sqrt((pix1[0] - pix2[0])**2 + (pix1[1] - pix2[1])**2 )


def disjunction(l1,l2,thresh = 5):
    res = []
    for pix1 in l1:
        for pix2 in l2:
            if dist(pix1, pix2) < thresh:
                res.append(np.mean(np.array([pix1,pix2]),axis = 0))
    return res


def border(img):
    tmp = np.zeros( (img.shape[0],img.shape[1]) )
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i > 0 and i < img.shape[0]-1 and j > 0 and j < img.shape[1]-1:
                tmp[i][j] = np.average([abs(img[i][j] - img[i+1][j])\
                                     , abs(img[i][j] - img[i-1][j])\
                                     , abs(img[i][j] - img[i][j+1])\
                                     , abs(img[i][j] - img[i][j-1])])
    return tmp


def avg(img):
    tmp = np.zeros( (img.shape[0],img.shape[1]) )
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
                if i > 0 and i < img.shape[0]-1 and j > 0 and j < img.shape[1]-1:
                    tmp[i][j] = np.average([img[i][j],\
                                            img[i+1][j],\
                                            img[i][j+1],\
                                            img[i-1][j],\
                                            img[i][j-1],\
                                            img[i-1][j-1],\
                                            img[i+1][j-1],\
                                            img[i-1][j+1],\
                                            img[i+1][j+1]])
    return tmp


def convert_to_color(img):
    tmp = np.zeros( (img.shape[0],img.shape[1],3 ))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            tmp[i][j][0] = img[i][j]
            tmp[i][j][1] = img[i][j]
            tmp[i][j][2] = img[i][j]
    return tmp


def _dfs(pix, img, new_cmt, mark):
    mark.append(pix)
    new_cmt.append(pix)
    dx = [0,0,-1,1]
    dy = [-1,1,0,0]
    stack = [pix]
    while(len(stack) > 0): 
        top = stack[-1]
        flg = False
        for i in range(4):
            x = top[0] + dx[i]
            y = top[1] + dy[i]
            if 0 < x and x < img.shape[0] and 0 < y and y < img.shape[1] and img[x][y] > 0 and [x,y] not in mark:
                flg = True
                top = [x,y]
                stack.append(top)
                mark.append(top)
                new_cmt.append(top)
                break
        if not flg:
            stack = stack[:-1]


def dfs(img):
    mark = []
    cmt = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > 0 and [i,j] not in mark:
                new_cmt = []
                _dfs([i,j], img, new_cmt, mark)
                if len(new_cmt) > 30:
                    cmt.append(new_cmt)
    return cmt


def draw_rect(img, ur, ll):
    for i in range(ur[0],ll[0]):
        img[i][ur[1]][0] = 255
        img[i][ur[1]][1] = 0
        img[i][ur[1]][2] = 0
        img[i][ll[1]][0] = 255
        img[i][ll[1]][1] = 0
        img[i][ll[1]][2] = 0
    for i in range(ur[1],ll[1]):
        img[ur[0]][i][0] = 255
        img[ur[0]][i][1] = 0
        img[ur[0]][i][2] = 0
        img[ll[0]][i][0] = 255
        img[ll[0]][i][1] = 0
        img[ll[0]][i][2] = 0
    return img


def get_boxes(img, regions):
    boxes = []
    for rgn in regions:
        ur_corner = np.min(rgn,axis =0)
        ll_corner = np.max(rgn,axis =0)
        boxes.append(img[ur_corner[0]:ll_corner[0],ur_corner[1]:ll_corner[1]])
    return boxes


def region_extract(img):
    #Threshold filter
    tmp = np.zeros( (img.shape[0],img.shape[1]) )
    img_cpy = np.zeros( (img.shape[0],img.shape[1]) )
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_cpy[i][j] = img[i][j]
    
    img = filter(img,40)

    regions = dfs(img)

    return regions,img_cpy,img


def extract_char(image_input):  # TODO
    regions, img_cpy,img= region_extract(image_input);
    box_img = get_boxes(img,regions)
    '''
    img = convert_to_color(img).astype(np.uint8)
    
    imgplot = plt.imshow(Image.fromarray(img))
    plt.show()
    '''
    return box_img
    

def resize(img):

    zeros = np.zeros((img.shape[0],1))
    while(img.shape[1] < 30):
        img = np.append(img,zeros,axis = 1)
        img = np.append(zeros,img,axis = 1)
    zeros = np.zeros((1,img.shape[1]))
    while(img.shape[0] < 30):
        img = np.append(img,zeros,axis = 0)
        img = np.append(zeros,img,axis = 0)

    tmp = img[int(img.shape[0]/2 - 14):int(img.shape[0]/2 + 14),int(img.shape[1]/2 - 14):int(img.shape[1]/2 + 14)]

    imgplot = plt.imshow(Image.fromarray(tmp),cmap='gray')
    plt.show()
    return tmp


def resize_imgs(img_list):
    List = []
    for img in img_list:
        List.append(resize(img))
    return List


def crack(image_input):
    img_list = extract_char(image_input)
    img_list = resize_imgs(img_list)

    classes = sess.run(classifier, feed_dict = {Input_image:img_list})
    
    image_input = ''
    for CHR in classes:
        image_input += chr(mapping[np.argmax(CHR)])

    return image_input


if __name__ == '__main__':
    #argument processing 
    parser = argparse.ArgumentParser( description='Train a CNN for EMNIST')
    parser.add_argument('command',help='Train a new network or evaluate the model')
    parser.add_argument('--model', required=False,help='Directory of the model weights')
    parser.add_argument('--train_data', required=False,help='Directory of the train data')
    parser.add_argument('--test_data', required=False,help='Directory of the test data')
    parser.add_argument('--image_input', required=False,help='image_input file')
    args = parser.parse_args()

    if args.command == "train":
        if args.model != None:
            initialize("LOAD",args.model)
        else:
            initialize("DEFAULT")
        Train(args.train_data,args.test_data)

    if args.command == "evaluate":
        initialize("LOAD",args.model)
        print("Accuracy : " + str(Test(args.test_data)))

    if args.command == "predict":
        mapping = get_mapping()
        initialize("LOAD",args.model)
        image_input_list = load_image_input(args.image_input)
        for image_input in image_input_list:
            print(crack(image_input))



