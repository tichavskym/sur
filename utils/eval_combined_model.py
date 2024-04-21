import argparse
import numpy as np

# The function for parsing the user arguments.
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train the models weights')    
    parser.add_argument('--resnet', type=str, required=True, help='Path to the file containing the ResNet18 evaluation on the test set.')
    parser.add_argument('--gmm', type=str, required=True, help='Path to the file containing the GMM evaluation on the test set.')
    parser.add_argument('--resnet_weight', type=float, default=0.5, help='The ResNet18 model trained weight for the combination.')
    parser.add_argument('--gmm_weight', type=float, default=0.5, help='The GMM model trained weight for the combination.')
    return parser.parse_args()

# Function for loading the data from the file in the required format.
def load_data_models(file_path):
    parsed_data = []

    with open(file_path, 'r') as file:
        for line in file:
            # Strip any leading/trailing whitespace and split the line into parts.
            parts = line.strip().split()
            
            if len(parts) >= 2:
                file_name = parts[0]
                prob = float(parts[1])

                parsed_data.append((file_name, prob))
            else:
                print(f"Error: invalid line format: {line}")
                break

    # Convert the list of tuples to a structured NumPy array.
    dtype = [('file_name', '<U20'), ('prob', float)]
    parsed_array = np.array(parsed_data, dtype=dtype)

    return parsed_array

def evaluate(gmm_data, resnet_data, gmm_weight, resnet_weight):
    # Validate that w1 + w2 = 1.
    assert abs(gmm_weight + resnet_weight - 1.0) < 1e-9, "Weights must sum to one."

    for gmm, resnet in zip(gmm_data, resnet_data):        
        # Check whether the files have the same file on the line.
        if gmm['file_name'] == resnet['file_name']:
            file_name = gmm['file_name']
            
            gmm_prob = gmm['prob']
            resnet_prob = resnet['prob']            

            prob = gmm_weight * gmm_prob + resnet_weight * resnet_prob

            pred = '1' if prob > 0.5 else '0'

            print(f'{file_name} {prob} {pred}')
        else:
            print(f"Error: invalid line format during training was detected.")
            break

if __name__ == "__main__":
    args = parse_arguments()

    # The files contain the evaluation at the target data as first 
    # (target-dev). As second the non-target data (non-target-dev).
    gmm_data = load_data_models(args.gmm)
    resnet_data = load_data_models(args.resnet)
    
    gmm_weight = args.gmm_weight
    resnet_weight = args.resnet_weight

    evaluate(gmm_data, resnet_data, gmm_weight, resnet_weight)
