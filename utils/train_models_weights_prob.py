import argparse
import numpy as np

# The function for parsing the user arguments.
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train the models weights')    
    parser.add_argument('--gt', type=str, required=True, help='Path to the file containing the ground truth.')
    parser.add_argument('--resnet', type=str, required=True, help='Path to the file containing the ResNet18 evaluation on the dev set.')
    parser.add_argument('--gmm', type=str, required=True, help='Path to the file containing the GMM evaluation on the dev set.')
    return parser.parse_args()

def load_data_ground_truth(file_path):
    parsed_data = []

    with open(file_path, 'r') as file:
        for line in file:
            # Strip any leading/trailing whitespace and split the line into parts.
            parts = line.strip().split()
            
            if len(parts) >= 2:
                file_name = parts[0]
                gt = int(parts[1])

                parsed_data.append((file_name, gt))
            else:
                print(f"Error: invalid line format: {line}")

    # Convert the list of tuples to a structured NumPy array.
    dtype = [('file_name', '<U20'), ('gt', int)]
    parsed_array = np.array(parsed_data, dtype=dtype)

    return parsed_array

# Function for loading the data from the file in the required format.
def load_data_models(file_path):
    parsed_data = []

    with open(file_path, 'r') as file:
        for line in file:
            # Strip any leading/trailing whitespace and split the line into parts.
            parts = line.strip().split()
            
            if len(parts) >= 3:
                file_name = parts[0]
                prob = float(parts[1])
                pred = int(parts[2])

                parsed_data.append((file_name, prob, pred))
            else:
                print(f"Error: invalid line format: {line}")
                break

    # Convert the list of tuples to a structured NumPy array.
    dtype = [('file_name', '<U20'), ('prob', float), ('pred', int)]
    parsed_array = np.array(parsed_data, dtype=dtype)

    return parsed_array

def convert_to_sum_1(value1, value2):
    total_sum = value1 + value2
    if total_sum == 0:
        return (0.5, 0.5)
    else:
        converted_value1 = value1 / total_sum
        converted_value2 = value2 / total_sum
        return converted_value1, converted_value2

def train(gt_data, gmm_data, resnet_data):    
    # Initial weight 0.0 for both models.
    gmm_weight = 0.0
    resnet_weight = 0.0

    for gt, gmm, resnet in zip(gt_data, gmm_data, resnet_data):
        # Check whether the files have the same file on the line.
        if gt['file_name'] == gmm['file_name'] == resnet['file_name']:
            gt_val = gt['gt']
            gmm_pred = gmm['pred']
            resnet_pred = resnet['pred']
            
            gmm_prob = gmm['prob']
            resnet_prob = resnet['prob']

            # Rewards and penalizations are counted from the 0.5 probability level.
            if gt_val == gmm_pred:
                if gt_val == 0:
                    # Correct, non-target.
                    # If the probability is for example 0.2, add reward 0.3.
                    gmm_weight = gmm_weight + (1 - gmm_prob) - 0.5
                else:
                    # Correct, target.
                    # If the probability is for example 0.8, add reward 0.3.
                    gmm_weight = gmm_weight + gmm_prob - 0.5
            else:
                if gt_val == 0:
                    # Incorrect, required non-target.
                    # If the probability is for example 0.8, penalize 0.3.
                    gmm_weight = gmm_weight - gmm_prob + 0.5
                else:
                    # Incorrect, required target.
                    # If the probability is for example 0.2, penalize 0.3.
                    gmm_weight = gmm_weight - (1 - gmm_prob) + 0.5
            
            if gt_val == resnet_pred:
                if gt_val == 0:
                    # Correct, non-target.
                    # If the probability is for example 0.2, add reward 0.3.
                    resnet_weight = resnet_weight + (1 - resnet_prob) - 0.5
                else:
                    # Correct, target.
                    # If the probability is for example 0.8, add reward 0.3.
                    resnet_weight = resnet_weight + resnet_prob - 0.5
            else:
                if gt_val == 0:
                    # Incorrect, required non-target.
                    # If the probability is for example 0.8, penalize 0.3.
                    resnet_weight = resnet_weight - resnet_prob + 0.5
                else:
                    # Incorrect, required target.
                    # If the probability is for example 0.2, penalize 0.3.
                    resnet_weight = resnet_weight - (1 - resnet_prob) + 0.5
        else:
            print(f"Error: invalid line format during training was detected.")
            break

    # Calculate proportions such that the sum of weights is 1.    
    proportional_gmm_weight, proportional_resnet_weight = convert_to_sum_1(gmm_weight, resnet_weight)

    return proportional_gmm_weight, proportional_resnet_weight

if __name__ == "__main__":
    args = parse_arguments()

    # The files contain the evaluation at the non-target data as first 
    # (non-target-dev). As second the target data (target-dev).
    gt_data = load_data_ground_truth(args.gt)    
    gmm_data = load_data_models(args.gmm)
    resnet_data = load_data_models(args.resnet)
    
    gmm_weight, resnet_weight = train(gt_data, gmm_data, resnet_data)

    print("GMM weight: ", gmm_weight)
    print("ResNet18 weight: ", resnet_weight)
