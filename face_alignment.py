import argparse
from argparse import Namespace

import os
import sys

import numpy as np
from PIL import Image

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


def main():
    parser = argparse.ArgumentParser(
        description='Align a square RGB face image',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--image_path',  help='Input path to face image, needs to be square and RGB', dest='image_path', required=True)
    parser.add_argument('--output_path',  help='Output file path, will be saved as a 256px x 256px image. JPG will work.', dest='output_path', required=True)
  
    # make sure output folder exists, otherwise saving won’t work
    if not os.path.exists('./face_alignment_outputs/'):
        os.makedirs('./face_alignment_outputs/')

    args = parser.parse_args()
    
    aligned_image = run_alignment(args.image_path)

    aligned_image.save(args.output_path)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

