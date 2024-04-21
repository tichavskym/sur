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

def train(gt_data, gmm_data, resnet_data):
    # The step size is calculated in a manner, that if the one model will be 
    # right in 100% of the examples and the second model will be wrong in 100% 
    # of the examples, then the first model will have weight=1 and the second 
    # weight=0. Based on that the behaviour will be, that the evaluation will 
    # be made only by the first model.
    step_size = 0.5 / float(len(gt_data))

    # Initial weight 0.5 for both models.
    gmm_weight = 0.5 # The resnet weight is calculated as (1 - gmm_weight).

    for gt, gmm, resnet in zip(gt_data, gmm_data, resnet_data):
        # Check whether the files have the same file on the line.
        if gt['file_name'] == gmm['file_name'] == resnet['file_name']:
            gt_val = gt['gt']
            gmm_pred = gmm['pred']
            resnet_pred = resnet['pred']
            if gt_val == gmm_pred and gt_val != resnet_pred:
                # The gmm was correct and resnet wrong. Increase the weight for the gmm model.
                gmm_weight = gmm_weight + step_size
            elif gt_val != gmm_pred and gt_val == resnet_pred:
                # The gmm was wrong and resnet correct. Decrease the weight for the gmm model.
                gmm_weight = gmm_weight - step_size
            # Ignore all other cases (both models were correct or wrong).
        else:
            print(f"Error: invalid line format during training was detected.")
            break
    
    return gmm_weight, 1 - gmm_weight

if __name__ == "__main__":
    args = parse_arguments()

    # The files contain the evaluation at the target data as first 
    # (target-dev). As second the non-target data (non-target-dev).
    gt_data = load_data_ground_truth(args.gt)    
    gmm_data = load_data_models(args.gmm)
    resnet_data = load_data_models(args.resnet)
    
    gmm_weight, resnet_weight = train(gt_data, gmm_data, resnet_data)

    print("GMM weight: ", gmm_weight)
    print("ResNet18 weight: ", resnet_weight)
