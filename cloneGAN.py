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

sys.path.append(".")
sys.path.append("..")

from utils.common import tensor2im
from models.psp import pSp
from models.e4e import e4e

# I don't think this is necessary
# load_ext autoreload
# %autoreload 2

# Input Image
# NUM_OUTPUT_IMAGES = 6
# image_path = 'notebooks/images/face_img.jpg'

def project(image_path: str, output_path: str, network: str, NUM_OUTPUT_IMAGES: int):
    experiment_type = 'ffhq_encode' #['ffhq_encode', 'cars_encode', 'church_encode', 'horse_encode', 'afhq_wild_encode', 'toonify']

    CODE_DIR = 'restyle-encoder'

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
        # "cars_encode": {"id": "1zJHqHRQ8NOnVohVVCGbeYMMr6PDhRpPR", "name": "restyle_psp_cars_encode.pt"},
        # "church_encode": {"id": "1bcxx7mw-1z7dzbJI_z7oGpWG1oQAvMaD", "name": "restyle_psp_church_encode.pt"},
        # "horse_encode": {"id": "19_sUpTYtJmhSAolKLm3VgI-ptYqd-hgY", "name": "restyle_e4e_horse_encode.pt"},
        # "afhq_wild_encode": {"id": "1GyFXVTNDUw3IIGHmGS71ChhJ1Rmslhk7", "name": "restyle_psp_afhq_wild_encode.pt"},
        # "toonify": {"id": "1GtudVDig59d4HJ_8bGEniz5huaTSGO_0", "name": "restyle_psp_toonify.pt"}
    }

    path = MODEL_PATHS[experiment_type]
    download_command = get_download_model_command(file_id=path["id"], file_name=path["name"]) 

    EXPERIMENT_DATA_ARGS = {
        "ffhq_encode": {
            "model_path": network, #"pretrained_models/restyle_psp_ffhq_encode.pt",
            # "image_path": "notebooks/images/face_img.jpg",
            "transform": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        },
        # "cars_encode": {
        #     "model_path": "pretrained_models/restyle_psp_cars_encode.pt",
        #     "image_path": "notebooks/images/car_img.jpg",
        #     "transform": transforms.Compose([
        #         transforms.Resize((192, 256)),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        # },
        # "church_encode": {
        #     "model_path": "pretrained_models/restyle_psp_church_encode.pt",
        #     "image_path": "notebooks/images/church_img.jpg",
        #     "transform": transforms.Compose([
        #         transforms.Resize((256, 256)),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        # },
        # "horse_encode": {
        #     "model_path": "pretrained_models/restyle_e4e_horse_encode.pt",
        #     "image_path": "notebooks/images/horse_img.jpg",
        #     "transform": transforms.Compose([
        #         transforms.Resize((256, 256)),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        # },
        # "afhq_wild_encode": {
        #     "model_path": "pretrained_models/restyle_psp_afhq_wild_encode.pt",
        #     "image_path": "notebooks/images/afhq_wild_img.jpg",
        #     "transform": transforms.Compose([
        #         transforms.Resize((256, 256)),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        # },
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


    # LOAD PRETRAINED MODEL

    model_path = EXPERIMENT_ARGS['model_path']
    ckpt = torch.load(model_path, map_location='cpu')

    opts = ckpt['opts']

    # update the training options
    opts['checkpoint_path'] = model_path

    opts = Namespace(**opts)
    if experiment_type == 'horse_encode': 
        net = e4e(opts)
    else:
        net = pSp(opts)
        
    net.eval()
    net.cuda()
    print('Model successfully loaded!')

    # VISUALIZE INPUT
    # image_path = EXPERIMENT_DATA_ARGS[experiment_type]["image_path"]
    original_image = Image.open(image_path).convert("RGB")

    if experiment_type == 'cars_encode':
        original_image = original_image.resize((192, 256))
    else:
        original_image = original_image.resize((256, 256))


    # ALIGN IMAGE
    def run_alignment(image_path):
        import dlib
        from scripts.align_faces_parallel import align_face
        if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
            print('Downloading files for aligning face image...')
            os.system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
            os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')
            print('Done.')
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        aligned_image = align_face(filepath=image_path, predictor=predictor) 
        print("Aligned image has shape: {}".format(aligned_image.size))
        return aligned_image 

    if experiment_type in ['ffhq_encode', 'toonify']:
        input_image = run_alignment(image_path)
    else:
        input_image = original_image

    # PERFORM INFERENCE
    img_transforms = EXPERIMENT_ARGS['transform']
    transformed_image = img_transforms(input_image)

    def get_avg_image(net):
        avg_image = net(net.latent_avg.unsqueeze(0),
                        input_code=True,
                        randomize_noise=False,
                        return_latents=False,
                        average_code=True)[0]
        avg_image = avg_image.to('cuda').float().detach()
        if experiment_type == "cars_encode":
            avg_image = avg_image[:, 32:224, :]
        return avg_image

    # number of output images
    opts.n_iters_per_batch = NUM_OUTPUT_IMAGES 
    opts.resize_outputs = False  # generate outputs at full resolution

    from utils.inference_utils import run_on_batch

    with torch.no_grad():
        avg_image = get_avg_image(net)
        tic = time.time()
        result_batch, result_latents = run_on_batch(transformed_image.unsqueeze(0).cuda(), net, opts, avg_image)
        toc = time.time()
        print('Inference took {:.4f} seconds.'.format(toc - tic))


    # VISUALIZE RESULT
    if opts.dataset_type == "cars_encode":
        resize_amount = (256, 192) if opts.resize_outputs else (512, 384)
    else:
        resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)

    def get_coupled_results(result_batch, transformed_image):
        """
        Visualize output images from left to right (the input image is on the right)
        """
        result_tensors = result_batch[0]  # there's one image in our batch
        result_images = [tensor2im(result_tensors[iter_idx]) for iter_idx in range(opts.n_iters_per_batch)]
        input_im = tensor2im(transformed_image)
        res = np.array(result_images[0].resize(resize_amount))
        for idx, result in enumerate(result_images[1:]):
            res = np.concatenate([res, np.array(result.resize(resize_amount))], axis=1)
        res = np.concatenate([res, input_im.resize(resize_amount)], axis=1)
        res = Image.fromarray(res)
        return res, result_images


    # get results & save
    res, result_images = get_coupled_results(result_batch, transformed_image)

    output_filename = os.path.splitext(os.path.basename(output_path))[0]
    output_extension = os.path.splitext(os.path.basename(output_path))[1]
    output_basedir = os.path.dirname(output_path)

    # SAVE OUT ORIG AS IMG
    image_filename = os.path.splitext(os.path.basename(image_path))[0]
    input_image.resize(resize_amount).save(f'{output_basedir}/{output_filename}_orig.jpg')

    # SAVE OUT EACH STEP AS AN IMG
    for idx, result in enumerate(result_images):
        outfile_path = f"{output_basedir}/{output_filename}_{idx}{output_extension}"
        Image.fromarray(np.array(result.resize(resize_amount))).save(outfile_path)

    # SAVE FINAL SUMMARY AS IMG
    res.save(f'{output_basedir}/{output_filename}_results.jpg')


def main():
    parser = argparse.ArgumentParser(
        description='Project given image.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--image_path',      help='Target image file to project to. Note: This MUST be an image of a human face. The face-alignment portion of this script will error out if there is no face recognized in the photo.', dest='image_path', required=True)
    parser.add_argument('--output_path',      help='Output FILE path. The output images will be saved with {0,1,2,3,4,5} appended to the filename.', dest='output_path', required=True)
    parser.add_argument('--network',      help='Path to the pretrained network file.', dest='network', required=False, default='pretrained_models/restyle_psp_ffhq_encode.pt')
    parser.add_argument('--NUM_OUTPUT_IMAGES', help='Number of output images / steps to take', type=int, default=6)

    args = parser.parse_args()

    # make sure output folder exists, otherwise saving won’t work
    output_basedir = os.path.dirname(args.output_path)
    if not os.path.exists(output_basedir):
        os.makedirs(output_basedir)

    project(**vars(parser.parse_args()))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

