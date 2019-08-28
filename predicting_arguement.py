import argparse


def invoke_arguement():
    parser = argparse.ArgumentParser(
        description="Image Classification.",
        usage="python ./predict.py/path/to/image.jpg my_checkpoint.pth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('path_to_image',
                        help='The path to the image file.',
                        action="store")

    parser.add_argument('checkpoint_file',
                        help='Path to checkpoint file.',
                        action="store")

    parser.add_argument('--save_dir',
                        action="store",
                        default=".",
                        dest='save_dir',
                        type=str,
                        help='The path to save the training checkpoint file',
                        )

    parser.add_argument('--top_k',
                        action="store",
                        default=5,
                        dest='top_k',
                        type=int,
                        help='Return most likely top_k classes.',
                        )

    parser.add_argument('--category_names',
                        action="store",
                        default="cat_to_name.json",
                        dest='categories_json',
                        type=str,
                        help='Path to file containing the categories.',
                        )

    parser.add_argument('--gpu',
                        action="store_true",
                        dest="use_gpu",
                        default=False,
                        help='Make sure you use gpu')

    parser.parse_args()
    return parser


def main():
    
    print(f'This is a command line utility for predict.py.\nYou can also try python train.py -h.')


if __name__ == '__main__':
    main()
