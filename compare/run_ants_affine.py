import ants
import time

class ants_reg():
    def __init__(self, config):
        # Initialize registration configuration

        # Set transform type based on dataset
        if config["dataset"] == "chaos":
            self.transform_type = "Rigid"
        elif config["dataset"] == "amos":
            self.transform_type = "Similarity"
        elif config["dataset"] == "3dircab" or config["dataset"] == "khp":
            self.transform_type = "Affine"
        else:
            # Raise error if dataset is not supported
            raise ValueError(f'dataset not supported! dataset={config["dataset"]}' )
    
    def registration(self, moving, fix):
        # Extract paths from input
        moving_path = moving[-1]
        fix_path = fix[-1]

        # Get image and label paths for moving and fixed images
        moving_image_path = moving_path["image"]
        moving_label_path = moving_path["label"]
        fix_image_path = fix_path["image"]
        fix_label_path = fix_path["label"]

        # Check for portalvein and venacava in moving_path and fix_path
        portalvein_moving = moving_path.get("portalvein")
        venacava_moving = moving_path.get("venacava")
        portalvein_fix = fix_path.get("portalvein")
        venacava_fix = fix_path.get("venacava")

        # Load images
        moving_image = ants.image_read(moving_image_path)  # Load moving image
        fixed_image = ants.image_read(fix_image_path)  # Load fixed image
        moving_label = ants.image_read(moving_label_path)  # Load moving label
        fix_label = ants.image_read(fix_label_path)  # Load fixed label

        # Perform registration
        start_time = time.perf_counter()  # Start timer for registration
        result_ants = ants.registration(fixed_image, moving_image,
                                         type_of_transform=self.transform_type,
                                         aff_metric="mattes",
                                         verbose=False)
        end_time = time.perf_counter()  # End timer for registration

        # Apply transforms to labels
        moved_label = ants.apply_transforms(fix_label, moving_label,
                                            transformlist=result_ants['fwdtransforms'],
                                            interpolator='genericLabel')
        
        # Convert moved_label to uint8
        moved_label = moved_label.astype('uint8')

        # Load and transform portalvein if it exists
        moved_portalvein = None
        if portalvein_moving:
            portalvein_image_moving = ants.image_read(portalvein_moving)  # Load moving portalvein image
            portalvein_image_fix = ants.image_read(portalvein_fix)  # Load fixed portalvein image
            moved_portalvein = ants.apply_transforms(portalvein_image_fix, portalvein_image_moving,
                                                                  transformlist=result_ants['fwdtransforms'],
                                                                  interpolator='linear')

        # Load and transform venacava if it exists
        moved_venacava = None
        if venacava_moving:
            venacava_image_moving = ants.image_read(venacava_moving)  # Load moving venacava image
            venacava_image_fix = ants.image_read(venacava_fix)
            moved_venacava = ants.apply_transforms(venacava_image_fix, venacava_image_moving,
                                                                transformlist=result_ants['fwdtransforms'],
                                                                interpolator='linear')

        # Prepare result dictionary with transformed outputs
        result = {
            "moved_image": result_ants['warpedmovout'],  # Transformed moving image
            "moved_label": moved_label,  # Transformed moving label
            "moved_portalvein": moved_portalvein,  # Transformed and thresholded portalvein (if available)
            "moved_venacava": moved_venacava,  # Transformed and thresholded venacava (if available)
            "affine_matrix": result_ants['fwdtransforms'][0],  # Affine matrix from registration
            "svf": None,  # Placeholder for stationary velocity field (not used here)
            "deformation": None,  # Placeholder for deformation field (not used here)
            "run_time": end_time - start_time  # Total runtime for the registration process
        }

        return result