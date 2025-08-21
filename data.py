import os
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np

class CustomDataset_Train(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.data_info = self._gather_data_info()

    def _gather_data_info(self):
        data_info = []
        for subfolder in ["skull/skull", "wheel/wheel", "plane/plane", "hand/hand"]:
            subfolder_path = os.path.join(self.folder_path, subfolder)
            for i in range(1, 51):  # 50 samples per category
                mat_files = {"inputs": {}, "fusions": {}, "ground_truth": ""}
                for algo in ["AIHT", "AMP", "FISTA", "Matched_filter", "ROMP"]:
                    mat_files["inputs"][algo] = [
                        os.path.join(subfolder_path, f"{algo}_img_{i}_angle_{angle}.mat")
                        for angle in range(1, 17)
                    ]
                    mat_files["fusions"][algo] = os.path.join(subfolder_path, f"{algo}_img_{i}_fusion.mat")
                mat_files["ground_truth"] = os.path.join(subfolder_path, f"Ground_Truth_img_{i}.mat")
                data_info.append(mat_files)
        return data_info

    def __len__(self):
        return len(self.data_info)

    def _replace_nan(self, matrix):
        return np.nan_to_num(matrix, nan=0.0)

    def __getitem__(self, idx):
        mat_files = self.data_info[idx]
        input_data = []
        fusion_data = []

        for algo in ["AIHT", "AMP", "FISTA", "Matched_filter", "ROMP"]:
            algo_data = []
            for angle_file in mat_files["inputs"][algo]:
                mat = sio.loadmat(angle_file)
                matrix = mat[list(mat.keys())[-1]]
                matrix = self._replace_nan(matrix)
                algo_data.append(matrix)
            input_data.append(np.stack(algo_data, axis=0))  # [16, H, W]

            fusion_mat = sio.loadmat(mat_files["fusions"][algo])
            fusion_matrix = fusion_mat[list(fusion_mat.keys())[-1]]
            fusion_matrix = self._replace_nan(fusion_matrix)
            fusion_data.append(fusion_matrix)  # [H, W]

        input_data = np.stack(input_data, axis=0)  # [5, 16, H, W]
        fusion_data = np.stack(fusion_data, axis=0)  # [5, H, W]

        gt = sio.loadmat(mat_files["ground_truth"])
        gt = gt[list(gt.keys())[-1]]
        gt = self._replace_nan(np.flipud(gt).copy())  # [H, W]
        
        input_data = torch.tensor(input_data).float()
        fusion_data = torch.tensor(fusion_data).float()
        ground_truth = torch.tensor(gt).float().view(1, 150, 200)

        return input_data, fusion_data, ground_truth


class CustomDataset_Test(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.data_info = self._gather_data_info()

    def _gather_data_info(self):
        """
        Collect file paths and their metadata.
        Each data sample contains:
        - 16 angle-view matrices for each algorithm
        - 1 fusion matrix for each algorithm
        - 1 ground truth matrix
        """
        data_info = []
        for subfolder in ["skull/skull", "wheel/wheel", "plane/plane", "hand/hand", "tree/tree","dragon/dragon"]:
            subfolder_path = os.path.join(self.folder_path, subfolder)
            for i in range(1, 6):  # 50 samples per folder
                mat_files = {"inputs": {}, "fusions": {}, "ground_truth": ""}
                
                # Collect 16 angle-view matrices and fusion matrix for each algorithm
                for algo in ["AIHT", "AMP", "FISTA", "Matched_filter", "ROMP"]:
                    mat_files["inputs"][algo] = []
                    for angle in range(1, 17):  # 16 angles
                        mat_files["inputs"][algo].append(
                            os.path.join(subfolder_path, f"{algo}_img_{i}_angle_{angle}.mat")
                        )
                    mat_files["fusions"][algo] = os.path.join(subfolder_path, f"{algo}_img_{i}_fusion.mat")
                
                # Ground truth file
                mat_files["ground_truth"] = os.path.join(subfolder_path, f"Ground_Truth_img_{i}.mat")
                data_info.append(mat_files)
        return data_info

    def __len__(self):
        """
        Return total number of data samples.
        """
        return len(self.data_info)

    def _replace_nan(self, matrix):
        """
        Replace NaN values in the matrix with 0.
        """
        return np.nan_to_num(matrix, nan=0.0)

    def __getitem__(self, idx):
        """
        Return a data sample by index, including:
        - 16 angle-view matrices for 5 algorithms
        - 5 fusion matrices
        - 1 ground truth matrix
        """
        mat_files = self.data_info[idx]
        input_data = []
        fusion_data = []

        # Load 16 angle-view matrices for each of the 5 algorithms
        for algo in ["AIHT", "AMP", "FISTA", "Matched_filter", "ROMP"]:
            algo_data = []
            for angle_file in mat_files["inputs"][algo]:
                mat = sio.loadmat(angle_file)
                matrix = mat[list(mat.keys())[-1]]
                matrix = self._replace_nan(matrix)
                algo_data.append(matrix)
            input_data.append(np.stack(algo_data, axis=0))  # [16, H, W]
        
            # Load fusion matrix for the algorithm
            fusion_mat = sio.loadmat(mat_files["fusions"][algo])
            fusion_matrix = fusion_mat[list(fusion_mat.keys())[-1]]
            fusion_matrix = self._replace_nan(fusion_matrix)
            fusion_data.append(fusion_matrix)  # [H, W]

        # Convert input data to torch tensor
        input_data = np.stack(input_data, axis=0)  # [5, 16, H, W]
        input_data = torch.tensor(input_data).float()

        # Convert fusion data to torch tensor
        fusion_data = np.stack(fusion_data, axis=0)  # [5, H, W]
        fusion_data = torch.tensor(fusion_data).float()

        # Load and convert ground truth
        ground_truth = sio.loadmat(mat_files["ground_truth"])
        ground_truth = ground_truth[list(ground_truth.keys())[-1]]
        ground_truth = self._replace_nan(ground_truth)
        ground_truth = torch.tensor(np.flipud(ground_truth).copy()).float().view(-1, 150, 200)

        return input_data, fusion_data, ground_truth
