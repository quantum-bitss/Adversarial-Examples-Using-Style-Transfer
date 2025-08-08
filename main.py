import argparse
import torch
from attack import run_attack

def parse_args():
    parser = argparse.ArgumentParser(description='Style Transfer Attack')
    
    # Attack parameters
    parser.add_argument('--target_label', type=int, default=498,
                        help='Target class label for the adversarial attack')
    parser.add_argument('--num_steps', type=int, default=2000,
                        help='Number of optimization steps')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for optimization')
    
    # Loss weights
    parser.add_argument('--content_weight', type=float, default=1,
                        help='Weight for content loss')
    parser.add_argument('--style_weight', type=float, default=10,
                        help='Weight for style loss')
    parser.add_argument('--adv_weight', type=float, default=10,
                        help='Weight for adversarial loss')
    parser.add_argument('--tv_weight', type=float, default=0.0,
                        help='Weight for adversarial loss')
    
    # Directories
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory to save results')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for computation')
    
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Running attack with target label: {args.target_label}")
    print(f"Using device: {args.device}")
    
    run_attack(args)
    
    print("Attack completed successfully!")

if __name__ == "__main__":
    main()
