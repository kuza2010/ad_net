import argparse
import json
import os

import numpy
import torch
from matplotlib import image as mImage

from cnn.versions.ADNet_v3 import ADNetV3

DEFAULT_MODEL_DIR = "pretrained_model"
DEFAULT_MODEL_NAME = "model.pt"

AD_NET_VERSIONS = {
    1: {
        'model': None,
        'description': '1th version'
    },
    2: {
        'model': None,
        'description': '2nd version'
    },
    3: {
        'model': ADNetV3(),
        'description': '3th version'
    }
}


def predict(pretrained_model_path,
            image_path,
            version):
    # do some checks
    if not os.path.exists(pretrained_model_path):
        print(f'Model not found, path {pretrained_model_path}')
        exit(-1)
    if not os.path.exists(image_path):
        print(f'Image not found, path {image_path}')
        exit(-1)
    if AD_NET_VERSIONS.get(version) is None:
        print(f'Version {version} does not exist. Available: [{list(AD_NET_VERSIONS.keys())}]')
        exit(-1)

    final_json = {}

    # load state
    net = AD_NET_VERSIONS.get(version)['model']
    loaded_state = torch.load(pretrained_model_path, map_location='cpu')['model_state']

    with torch.no_grad():
        net.load_state_dict(loaded_state)
        net.eval()

        if os.path.isdir(args.image):
            images = os.listdir(args.image)
            for idx, val in enumerate(images):
                output = proceed(image_path + '/' + val, net)
                interpret_prediction(output, val, final_json, idx)
        else:
            output = proceed(image_path, net)
            interpret_prediction(output, image_path, final_json)

    print(json.dumps(final_json))


def proceed(path_to_image,
            net):
    img = mImage.imread(path_to_image, 'pgm')
    img_array = numpy.asarray(img, dtype='float32')
    img_array = numpy.expand_dims(img_array, axis=0)
    img_array = numpy.expand_dims(img_array, axis=0)
    img_array = torch.from_numpy(img_array)
    output = net(img_array)

    return output


def interpret_prediction(output,
                         image_name,
                         history,
                         idx=0):
    prediction = output.max(1, keepdim=True)
    prediction_tensor = prediction[1]

    cover_percentage, stego_percentage = output[0].numpy()
    if prediction_tensor[0].numpy()[0] == 0:
        history[idx] = {
            'imageName': os.path.basename(image_name),
            'probability': f'{cover_percentage:.4f}/{stego_percentage:.4f}',
            'result': 'STEGO'
        }
        # print(f'{idx + 1}) Image {image_name}\t[{cover_percentage:.4f}/{stego_percentage:.4f}] \tCOVER')
    else:
        history[idx] = {
            'imageName': os.path.basename(image_name),
            'probability': f'{cover_percentage:.4f}/{stego_percentage:.4f}',
            'result': 'STEGO'
        }
        # print(f'{idx + 1}) Image {image_name}\t[{cover_percentage:.4f}/{stego_percentage:.4f}] \tSTEGO')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Get prediction for provided image.',
        usage="python ad_predict.py --image ./stego_lena.pgm",
    )
    parser.add_argument(
        '-i', '--image',
        help='path to image to check',
        required=True
    )
    parser.add_argument(
        '-v', '--version',
        help='ADNet cnn version',
        required=False,
        default=max(list(AD_NET_VERSIONS.keys()))
    )
    parser.add_argument(
        '-m', '--model_path',
        help='path to model',
        required=False,
        default=DEFAULT_MODEL_DIR + '/' + DEFAULT_MODEL_NAME
    )

    args = parser.parse_args()
    predict(args.model_path, args.image, args.version)
