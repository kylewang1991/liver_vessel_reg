import ants
import time
class sync_reg():
    def __init__(self, config) -> None:
        pass

    def registration(self, moving, fix):
        moving_path = moving[-1]
        fix_path = fix[-1]

        # Get image and label paths for moving and fixed images
        moving_image_path = moving_path["image"]
        moving_label_path = moving_path["label"]
        fix_image_path = fix_path["image"]
        fix_label_path = fix_path["label"]

        # Check for portalvein and venacava in moving_path
        portalvein_moving = moving_path.get("portalvein")
        venacava_moving = moving_path.get("venacava")
        portalvein_fix = fix_path.get("portalvein")
        venacava_fix = fix_path.get("venacava")

        # Load images
        moving_image = ants.image_read(moving_image_path)  # Load moving image
        fixed_image = ants.image_read(fix_image_path)  # Load fixed image
        moving_label = ants.image_read(moving_label_path)  # Load moving label
        fix_label = ants.image_read(fix_label_path)  # Load fixed label

        start_time = time.perf_counter()

        result_ants = ants.registration(fixed_image, moving_image,
                                type_of_transform="SyN",
                                verbose=False)
        
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
            
          
        
        end_time = time.perf_counter()

        result = {}
        result["moved_image"] = result_ants['warpedmovout']
        result["moved_label"] = moved_label
        result["moved_portalvein"] = moved_portalvein
        result["moved_venacava"] = moved_venacava
        result["affine_matrix"] = result_ants['fwdtransforms'][1]
        result["svf"] = result_ants['fwdtransforms'][0]
        result["deformation"] = None
        result["run_time"] = end_time - start_time

        return result
