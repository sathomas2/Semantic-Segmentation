from glob import glob
import os, argparse
import tensorflow as tf
import numpy as np
import scipy.misc
import cv2

import helper
import labels



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segment test set')
    parser.add_argument(
        'dataset',
        type=str,
        nargs='?',
        default='KITTI',
        help='type cityscapes or KITTI.'
    )
    parser.add_argument(
        'data_dir',
        type=str,
        nargs='?',
        default='/Volumes/Seagate/Code/udacity_SDC/term3/CarND-Semantic-Segmentation/data/testing',
        help='type directory where test data is stored.'
    )
    parser.add_argument(
        'check_dir',
        type=str,
        nargs='?',
        default='checks_KITTI',
        help='Path to model checkpoints to be frozen and used for inference.'
    )
    args = parser.parse_args()



def freeze_graph(model_dir=args.check_dir, output_node_names='preds'):
    """Extract the sub graph defined by the output nodes and convert 
    all its variables into constant 
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names, 
                            comma separated
    """
    ### CITATION: https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
    
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    #input_checkpoint = model_dir + '/' + model
    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)
    
        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        ) 

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
    
def load_graph(frozen_graph_filename=args.check_dir+'/frozen_model.pb'):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def save_test_imgs(x, pred, num_classes, fn, save_dir, orig_shape=(375, 1242), KITTI=True):
    if KITTI: 
        preds_frozen = np.copy(pred)
        pred = np.zeros((preds_frozen.shape[0], preds_frozen.shape[1], 30))
        pred0 = np.ones_like(preds_frozen[:,:,0]) - preds_frozen[:,:,0]
        pred[:,:,0] = pred0
        pred[:,:,3] = preds_frozen[:,:,0]
        
    temp_pred = np.zeros((pred.shape[0], pred.shape[1], 3))
    
    for i in range(num_classes):
        color = labels.id2color[i]
        pred_arg = np.argmax(pred, axis=-1)
        temp_pred[:,:,0][pred_arg == i] = color[0] / 255.
        temp_pred[:,:,1][pred_arg == i] = color[1] / 255.
        temp_pred[:,:,2][pred_arg == i] = color[2] / 255.
        
    img = scipy.misc.imresize(x, orig_shape)
    pred_img = scipy.misc.imresize(temp_pred, orig_shape)
    new_img = cv2.addWeighted(img, 1, pred_img, 0.3, 0)
    
    scipy.misc.imsave(save_dir+'/'+fn.split('/')[-1].split('.')[0]+'_prediction.png', pred_img)
    scipy.misc.imsave(save_dir+'/'+fn.split('/')[-1].split('.')[0]+'_overlay.png', new_img)
    
    return 0

freeze_graph()

data_folder = args.data_dir
if args.dataset == 'KITTI':
    orig_shape = (375, 1242)
    img_shape = (256, 856)
    save_dir = 'run/KITTI_results'
    kitty = True
    mask_shape = img_shape
    get_batches_fn = helper.gen_batch_function_KITTI(data_folder=data_folder, image_shape=img_shape, 
                                                     mask_shape=img_shape, num_classes=2, mode='test')
else:
    orig_shape = (1024, 2048)
    img_shape = (256, 512)
    save_dir = 'run/cityscapes_results'
    kitty = False
    mask_shape = img_shape
    #data_folder='/home/ubuntu/CarND-Semantic-Segmentation/data/cityscapes'
    get_batches_fn = helper.gen_batch_function(data_folder=data_folder,image_shape=img_shape, 
                                               mask_shape=mask_shape, num_classes=30)


batch_size = 10


tf.reset_default_graph()
graph = load_graph(frozen_graph_filename=args.check_dir+'/frozen_model.pb')

with tf.Session(graph=graph) as sess:
    x = graph.get_tensor_by_name("prefix/x:0")
    train_mode = graph.get_tensor_by_name("prefix/train_mode:0")
    preds = graph.get_tensor_by_name("prefix/preds:0")
    
    #cnt = 0
    for batch_x, batch_y, _ in get_batches_fn(batch_size, mode='test'):
        feed_dict ={x: batch_x,
                train_mode: False}
        preds_frozen = sess.run(preds,feed_dict)
    
        _ = [save_test_imgs(batch_x[i], preds_frozen[i], 30, batch_y[i], save_dir, orig_shape=orig_shape, KITTI=kitty) 
         for i in range(len(batch_y))]
        #cnt += 1
        #if cnt > 0: 
            #break