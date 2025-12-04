import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as R

class random_affine_reg():
    def __init__(self, config) -> None:
        pass

    @staticmethod
    def compose_linear_transform(angle, scale, translation):
        # Create rotation object
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = R.from_euler('xyz', angle, degrees=True).as_matrix()

        # Create scaling matrix
        scaling_matrix = np.diag(np.append(scale, 1))
            
        # Create translation matrix
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = translation
            
        # Compose the transformation matrix
        transformation_matrix = translation_matrix @ rotation_matrix @ scaling_matrix
            
        return transformation_matrix

    def random_linear_transform(self, image, label):
            
        angle = np.random.uniform(-10, 10, size=3)
        scale  = np.random.uniform(0.9, 1.1, size=3)
        translation = np.random.uniform(-0.1, 0.1, size=3)

        # angle = np.zeros(3)
        # scale  = np.array([1.5, 1.5, 1.125])
        # translation = np.zeros(3)
        
        print(f"angle: {angle}")
        print(f"scale: {scale}")
        print(f"translation: {translation}")
        
        mat = self.compose_linear_transform(angle, scale, translation)
        mat = torch.from_numpy(mat).unsqueeze(0)
        mat = mat[:, :3, :]

        grid = F.affine_grid(mat, [1, 3, image.shape[2], image.shape[3], image.shape[4]], align_corners=True)
        
        transformed_image = F.grid_sample(image.float(), grid.float(), align_corners=True, mode='bilinear')
        transformed_label = F.grid_sample(label.float(), grid.float(), align_corners=True, mode='nearest')


        return transformed_image.to(torch.float), transformed_label.to(torch.uint8)

    def registration(self, moving, fix):

        moving_image = moving[0].unsqueeze(0).unsqueeze(0)
        moving_label = moving[1].unsqueeze(0).unsqueeze(0)
        moved_image, moved_label = self.random_linear_transform(moving_image, moving_label)

        result = {}
        result["moved_image"] = moved_image
        result["moved_label"] = moved_label
        result["affine_matrix"] = None
        result["svf"] = None
        result["deformation"] = None

        return result