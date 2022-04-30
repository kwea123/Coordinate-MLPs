import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path', type=str, default='images/fox.jpg',
                        help='path to the image to reconstruct')
    parser.add_argument('--arch', type=str, default='identity',
                        choices=['identity', 'pe', 'gau', 'siren'],
                        help='network structure')
    parser.add_argument('--sc', type=float, default=10.,
                        help='gaussian embedding scale factor')

    parser.add_argument('--batch_size', type=int, default=256*256,
                        help='number of batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='number of epochs')

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')

    return parser.parse_args()