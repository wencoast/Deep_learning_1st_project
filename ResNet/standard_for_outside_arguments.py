#def parse_args(): 
# we are going to rename this previous function in order to avoid confusion with bulit-in method. 
def parse_flags():

    parser = argparse.ArgumentParser() # Define one parser first
    parser.add_argument('--config_path', type=str, help='path to config file',
                        default='./configs/config_ms1m_100.yaml')
    return parser.parse_args() # parse_args should be the bulit-in method, we are going to call it. 

if __name__ == '__main__':
#    args = parse_args()
    arguments = parse_flags()

"""

One more style

"""

def parse_flags(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='path to config file',
                        default='./configs/config_ms1m_100.yaml')
    return parser.parse_args(argv)
# sys.argv[1:] is special.

def main(args):
    

if __name__ == '__main__':
    main(parse_flags(sys.argv[1:]))
# Here there has no any problem, since args is same to parse_flags(sys.argv[1:])

