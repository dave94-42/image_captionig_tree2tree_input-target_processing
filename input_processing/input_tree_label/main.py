from alexNet import *
import sys
import random

n_err=0


def label_tree_with_data(alex_image, alex_map,x, image_map, tree):

    """
    function used to label tree structure previously build with cnn data
    :param alex_image: cnn used for image i.e. alex_net until 3rd conv layer
    :param alex_map: cnn use for map i.e. same as above but convoltions go grom one channel to one channel
    :param x: real image
    :param image_map: map
    :param tree: previously built tree
    :return:
    """

    #real_image -> alex net
    y = alex_image.forward(x)
    n_segment = int(np.amax(image_map))
    nodes_data = []
    for i in range(1,n_segment+1):
        #for each segment buiild a "segmantat map" from image map
        seg = np.where(image_map==i,image_map,0 )
        x_seg=torch.from_numpy(seg)
        x_seg.unsqueeze_(0)
        x_seg.unsqueeze_(0)
        # segment map -> alex net
        y_seg = alex_map.forward(x_seg)
        non_zero = (y_seg!=0).nonzero()

        assert non_zero.sum()!=0
        assert ( y.shape[2] == y_seg.shape[2] and y.shape[3] == y_seg.shape[3] )

        #node_data = torch.zeros(256)
        node_data = torch.ones(256)*np.finfo(float).min
        for j in range(1,non_zero.shape[0]):
            #for each element that is non zero take all channels and compute mean
            dim1 = int(non_zero[j][2])
            dim2 = int(non_zero[j][3])
            element = y[0,:,dim1,dim2]
            #node_data += element
            node_data = torch.max(node_data,element)
        #node_data /= non_zero.shape[0]
        nodes_data.append(node_data)

    assert len(nodes_data) == n_segment

    tree.write_data_in_leafs(nodes_data)


def write_on_file(file_name,dataset):
    """
    function to write tree, labled wit cnn_info, in file
    """
    f=open( file_name, "w+")

    for data in dataset:
        tree = data["tree"]
        name = data["name"]
        string=tree.get_string_from_tree()
        f.write( name +" : ("+string[:-1]+");\n")



def main ():
    #check arguments
    args = sys.argv[1:]
    if len(args) <1 or args[0]=="--help":
        print("arguments:\n"
              "1)final glia dir (i.e. dir containing files describing trees and segment map files\n"
              "2)images dir\n"
              "3)file to generate")
        exit()

    #load data
    data = myDataset(args[0],args[1])
    dataSet = []
    for i in range (0,data.__len__()):
        print("load" , i, "-th images" )
        x = data.__getitem__(i)
        dataSet.append(x)

    #instanciate myAlex
    alex_image = MyAlexNet()
    alex_map = SimpleCNN()

    i=0
    for data in dataSet:
        #for each image in data take image, mape, and tree
        x = data["image"]
        map = data["map"]
        tree = data["tree"]

        label_tree_with_data(alex_image, alex_map,x, map, tree)
        tree.check_tree()
        i=i+1
        print ("processed ",i,"-th images")

    #write on specified file
    print ("writing result on disk")
    write_on_file(args[2],dataSet)



if __name__ == '__main__':
    main()
