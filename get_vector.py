import argparse
from argparse import Namespace
import time
import os
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import json

sys.path.append(".")
sys.path.append("..")

from utils.common import tensor2im
from models.psp import pSp

from utils.inference_utils import run_on_batch

resize_amount = (1024,1024)
experiment_type = 'ffhq_encode' #['ffhq_encode', 'toonify']
CODE_DIR = 'restyle-encoder'
NUM_STEPS = 6

# SAVE OUT OUTPUT IMAGES
def saveResults(result_images, output_path):

    output_filename = 'out'
    output_extension ='jpg'
    output_basedir = output_path

    # make sure output folder exists, otherwise saving won’t work
    if not os.path.exists(output_basedir):
        os.makedirs(output_basedir)

    # SAVE OUT ORIG AS IMG
    # image_filename = os.path.splitext(os.path.basename(image_path))[0]
    # input_image.resize(resize_amount).save(f'{output_basedir}/{output_filename}_orig.jpg')

    # SAVE OUT EACH STEP AS AN IMG
    for idx, result in enumerate(result_images):
        outfile_path = f"{output_basedir}/{output_filename}_{idx}.{output_extension}"
        Image.fromarray(np.array(result.resize(resize_amount))).save(outfile_path)

    # SAVE FINAL SUMMARY AS IMG
    # res.save(f'{output_basedir}/{output_filename}_results.jpg')


# def main():
#     parser = argparse.ArgumentParser(
#         description='Project given image.',
#         formatter_class=argparse.RawDescriptionHelpFormatter
#     )

#     parser.add_argument('--image_path',      help='Target image file to project to. Note: This MUST be an image of a human face. The face-alignment portion of this script will error out if there is no face recognized in the photo.', dest='image_path', required=True)
#     parser.add_argument('--output_path',      help='Output FILE path. The output images will be saved with {0,1,2,3,4,5} appended to the filename.', dest='output_path', required=True)
#     parser.add_argument('--network',      help='Path to the pretrained network file.', dest='network', required=False, default='pretrained_models/restyle_psp_ffhq_encode.pt')
#     parser.add_argument('--NUM_OUTPUT_IMAGES', help='Number of output images / steps to take', type=int, default=6)

#     args = parser.parse_args()

#     # PROJECT
#     res, result_images = project(**vars(parser.parse_args()))   

#     # SAVE OUT
#     saveResults(result_images, args.output_path)

def get_avg_image(net):
    avg_image = net(net.latent_avg.unsqueeze(0),
                    input_code=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)[0]
    avg_image = avg_image.to('cuda').float().detach()
    return avg_image


    # MOVE FUNCTION TO DISPLAY RESULTS OUTSIDE OF LOOP
def get_coupled_results(result_batch, transformed_image):
    """
    Visualize output images from left to right (the input image is on the right)
    """
    result_tensors = result_batch[0]  # there's one image in our batch
    result_images = [tensor2im(result_tensors[iter_idx]) for iter_idx in range(NUM_STEPS)]
    input_im = tensor2im(transformed_image)
    res = np.array(result_images[0].resize(resize_amount))
    for idx, result in enumerate(result_images[1:]):
        res = np.concatenate([res, np.array(result.resize(resize_amount))], axis=1)
    res = np.concatenate([res, input_im.resize(resize_amount)], axis=1)
    res = Image.fromarray(res)
    return res, result_images

def parse_json(json_path):
    with open(json_path) as f:
      json_dict = json.load(f)
    return json_dict


def initialize(network_path, verbose):

    def get_download_model_command(file_id, file_name):
        """ Get wget download command for downloading the desired model and save to directory ../pretrained_models. """
        current_directory = os.getcwd()
        save_path = os.path.join(os.path.dirname(current_directory), CODE_DIR, "pretrained_models")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        url = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=save_path)
        return url    

    MODEL_PATHS = {
        "ffhq_encode": {"id": "1sw6I2lRIB0MpuJkpc8F5BJiSZrc0hjfE", "name": "restyle_psp_ffhq_encode.pt"},
        # "toonify": {"id": "1GtudVDig59d4HJ_8bGEniz5huaTSGO_0", "name": "restyle_psp_toonify.pt"}
    }

    path = MODEL_PATHS[experiment_type]
    download_command = get_download_model_command(file_id=path["id"], file_name=path["name"]) 

    EXPERIMENT_DATA_ARGS = {
        "ffhq_encode": {
            "model_path": network_path, #"pretrained_models/restyle_psp_ffhq_encode.pt",
            # "image_path": "notebooks/images/face_img.jpg",
            "transform": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        },
        # "toonify": {
        #     "model_path": "pretrained_models/restyle_psp_toonify.pt",
        #     "image_path": "notebooks/images/toonify_img.jpg",
        #     "transform": transforms.Compose([
        #         transforms.Resize((256, 256)),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        # },
    }

    EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]

    if not os.path.exists(EXPERIMENT_ARGS['model_path']) or os.path.getsize(EXPERIMENT_ARGS['model_path']) < 1000000:
        print(f'Downloading ReStyle model for {experiment_type}...')
        os.system(f"wget {download_command}")
        # if google drive receives too many requests, we'll reach the quota limit and be unable to download the model
        if os.path.getsize(EXPERIMENT_ARGS['model_path']) < 1000000:
            raise ValueError("Pretrained model was unable to be downloaded correctly!")
        else:
            print('Done.')
    else:
        print(f'ReStyle model for {experiment_type} already exists!')


    time_before = time.time()

    # LOAD PRETRAINED MODEL
    model_path = EXPERIMENT_ARGS['model_path']
    ckpt = torch.load(model_path, map_location='cpu')

    opts = ckpt['opts']

    # resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)

    # number of output images
    opts['n_iters_per_batch'] = NUM_STEPS
    opts['resize_outputs'] = False  # generate outputs at full resolution

    # update the training options
    opts['checkpoint_path'] = model_path

    opts = Namespace(**opts)
    net = pSp(opts)
        
    net.eval()
    net.cuda()
    print('Model successfully loaded!')

    time_after_loading = time.time()
    print('Time to load model took {:.4f} seconds.'.format(time_after_loading - time_before))


    if verbose:
        print(f'Loaded network from {network_path}.')

    return net, opts

def process(input_path, output_path, verbose, preloaded_params):
    time_before = time.time()

    image_path = input_path
    net = preloaded_params[0]
    opts = preloaded_params[1]

    # VISUALIZE INPUT
    # image_path = EXPERIMENT_DATA_ARGS[experiment_type]["image_path"]
    original_image = Image.open(image_path).convert("RGB")

    # ALIGN IMAGE
    # def run_alignment(image_path):
    #     import dlib
    #     from scripts.align_faces_parallel import align_face
    #     if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
    #         print('Downloading files for aligning face image...')
    #         os.system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
    #         os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')
    #         print('Done.')
    #     predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    #     aligned_image = align_face(filepath=image_path, predictor=predictor) 
    #     print("Aligned image has shape: {}".format(aligned_image.size))
    #     return aligned_image 

    # if experiment_type in ['ffhq_encode', 'toonify']:
    #     input_image = run_alignment(image_path)
    # else:
    #     input_image = original_image
    input_image = original_image

    # PERFORM INFERENCE
    img_transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    transformed_image = img_transforms(input_image)

    with torch.no_grad():
        avg_image = get_avg_image(net)
        tic = time.time()
        result_batch, result_latents = run_on_batch(transformed_image.unsqueeze(0).cuda(), net, opts, avg_image)
        toc = time.time()
        if verbose:
            print('Inference took {:.4f} seconds.'.format(toc - tic))


    # VISUALIZE RESULT

    # get results & save
    res, result_images = get_coupled_results(result_batch, transformed_image)

    time_after = time.time()
    if verbose:
        print('Time to load img and get its w-vector took {:.4f} seconds.'.format(time_after - time_before))

#     # SAVE OUT
    saveResults(result_images, output_path)
    # return res, result_images



def doLoop(preloaded_params, json_path, sleep_time, verbose):
        
    while True:

        if os.path.exists(json_path):
            # parse json
            json_dict = parse_json(json_path)

            # del json

# ADD THIS BACK IN LATER TOOOOOO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            # if verbose:
            #     print(f'Deleting JSON.')
            # os.remove(json_path)

            process(json_dict['input_path'], json_dict['output_path'], verbose, preloaded_params)

        else:
            SLEEP_TIME_IN_SECS = sleep_time/1000.0
            if verbose:
                print(f'JSON not found. Sleeping {SLEEP_TIME_IN_SECS} seconds.')
            time.sleep(SLEEP_TIME_IN_SECS)



def start(preloaded_params, json_path, sleep_time, verbose):
    doLoop(preloaded_params, json_path, sleep_time, verbose)

# ADD THIS BACK IN LATER TOOOOOO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # try:
    #     doLoop(preloaded_params, json_path, sleep_time, verbose)
    # except:
    #     if verbose:
    #         print(f'Exception thrown during loop. Re-entering loop.')
    #     start(preloaded_params, json_path, sleep_time, verbose)


def main():

    parser = argparse.ArgumentParser(
        description='Return the w-vector of the inputted image.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--network_path',     help='Network filepath as a .pt file', dest='network_path', required=True)
    parser.add_argument('--json_path',      help='File path to json arg file', dest='json_path', required=True)
    parser.add_argument('--sleep_time',      help='Sleep time in milliseconds', dest='sleep_time', type=int, default=50)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()


    preloaded_params = initialize(args.network_path, args.verbose)
    start(preloaded_params, args.json_path, args.sleep_time, args.verbose)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------