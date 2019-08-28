import argparse

'''
Required Architectures

'''
required_architecture = [
    'vgg11',
    'vgg13',
    'vgg16',
    'vgg19',
    'densenet121',
    'densenet161',
    'densenet169',
    'densenet201'
]


def invoke_arguement():
   

    parser = argparse.ArgumentParser(
        description="Training and saving an image classfier.",
        usage="python ./train.py ./flowers/train --gpu --learning_rate 0.001 --hidden_layers 3136 --epochs 5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('data_directory', action="store")

    parser.add_argument('--save_dir',
                        action="store",
                        default=".",
                        dest='save_dir',
                        type=str,
                        help='The path to save the training checkpoint file',
                        )

    parser.add_argument('--save_name',
                        action="store",
                        default="my_checkpoint",
                        dest='save_name',
                        type=str,
                        help='Checkpoint name.',
                        )

    parser.add_argument('--categories_json',
                        action="store",
                        default="cat_to_name.json",
                        dest='categories_json',
                        type=str,
                        help='Path to file containing the categories.',
                        )

    parser.add_argument('--arch',
                        action="store",
                        default="vgg16",
                        dest='arch',
                        type=str,
                        help='Supported architectures: ' + ", ".join(required_architecture),
                        )

    parser.add_argument('--gpu',
                        action="store_true",
                        dest="use_gpu",
                        default=False,
                        help='Use GPU')

    hyp = parser.add_argument_group('hyperparameters')

    hyp.add_argument('--learning_rate',
                    action="store",
                    default=0.001,
                    type=float,
                    help='The learning rate')

    hyp.add_argument('--hidden_layers', '-hu',
                    action="store",
                    dest="hidden_layers",
                    default=[3136, 784],
                    type=int,
                    nargs='+',
                    help='The hidden layers')

    hyp.add_argument('--epochs',
                    action="store",
                    dest="epochs",
                    default=1,
                    type=int,
                    help='Epochs')

    parser.parse_args()
    return parser


def main():
    
    print(f' This is a  command line utility for train.py.\nYou can also try python train.py -h.')


if __name__ == '__main__':
    main()
