import numpy as np
import warnings
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tabpfnwide.load_mm_data import load_multiomics
from tabpfn.model.loading import load_model_criterion_config
from tabpfn.model.memory import MemoryUsageEstimator
import warnings
warnings.filterwarnings("ignore")
import argparse


def main(dataset_name, output_file, checkpoint_path, device="cuda:0", omic="mrna"):
    """
    Extracts and saves attention maps from a trained transformer-based model on multi-omics data.
    Parameters:
        dataset_name (str): Name of the multi-omics dataset to use.
        output_file (str): Path to save the extracted attention maps (as a torch file).
        checkpoint_path (str): Path to the model checkpoint to load weights from.
        device (str, optional): Device to run the model on (default: "cuda:0").
        omic (str, optional): Omics data type to use from the dataset (default: "mrna").
    Description:
        - Loads multi-omics data and encodes labels.
        - Loads a TabPFN-Wide model.
        - Configures the model to save attention maps during inference.
        - Runs inference to obtain predictions and attention maps.
        - Saves the extracted attention maps to the specified output file.
    """
    
    ds_dict, labels = load_multiomics(dataset_name)
    mrna = ds_dict[omic]
    X, y = mrna.values, labels
    y = LabelEncoder().fit_transform(y)
    print(X.shape)

    model, _, _ = load_model_criterion_config(
            model_path=None,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=False,
            which='classifier',
            version='v2',
            download=True,
    )
    # Disable feature grouping
    model.features_per_group = 1

    checkpoint_path = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint_path["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.int8).unsqueeze(1).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int8).unsqueeze(1).to(device)

    for layer in model.transformer_encoder.layers:
        layer.self_attn_between_features.save_att_map = True
        layer.self_attn_between_features.number_of_samples = X_train_tensor.shape[0]

    MemoryUsageEstimator.SAVE_PEAK_MEM_FACTOR = 4 # Increase if less memory is available
    MemoryUsageEstimator.reset_peak_memory_if_required(
        save_peak_mem=True,
        model=model,
        X=torch.cat([X_train_tensor, X_test_tensor], dim=0),
        cache_kv=False,
        dtype_byte_size=2,
        device=torch.device(device),
    )

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred_logits = model(
                train_x=X_train_tensor,
                train_y=y_train_tensor,
                test_x=X_test_tensor,
            )
            n_classes = len(np.unique(y_train_tensor.cpu()))
            pred_logits = pred_logits[..., :n_classes].float()  
            pred_probs = torch.softmax(pred_logits, dim=-1)[:, 0, :].detach().cpu().numpy()


    atts = [getattr(layer.get_submodule("self_attn_between_features"), "attention_map") for layer in model.transformer_encoder.layers]
    atts = torch.stack(atts, dim=0)
    torch.save(atts, f"{output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to use")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the attention maps")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    parser.add_argument("--omic", type=str, default="mrna", help="Omic type to use (default: mRNA)")
    
    args = parser.parse_args()
    main(args.dataset_name, args.output_file, args.checkpoint_path, args.device, args.omic)
    


