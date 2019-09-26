def parse_args():
    
    parser = argparse.ArgumentParser() # Define one parser first
    parser.add_argument('--config_path', type=str, help='path to config file',
                        default='./configs/config_ms1m_100.yaml')
    return parser.parse_args() # parse_args should be the bulit-in method, we are going to call it. 

if __name__ == '__main__':
    args = parse_args()
