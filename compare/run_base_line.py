import SimpleITK as sitk

class baseline_reg():
    def __init__(self, config) -> None:
        pass

    def registration(self, moving, fix):
        # Extract paths from input
        moving_path = moving[-1]

        # Get image and label paths for moving images
        moving_image_path = moving_path["image"]
        moving_label_path = moving_path["label"]

        # Check for portalvein and venacava in moving_path
        portalvein_moving = moving_path.get("portalvein")
        venacava_moving = moving_path.get("venacava")

        # Load images using ANTsPy
        moving_image = sitk.ReadImage(moving_image_path)  # Load moving image
        moving_label = sitk.ReadImage(moving_label_path)  # Load moving label

        # Load portalvein and venacava if they exist
        moving_portalvein = None
        if portalvein_moving:
            moving_portalvein = sitk.ReadImage(portalvein_moving)  # Load moving portalvein label

        moving_venacava = None
        if venacava_moving:
            moving_venacava = sitk.ReadImage(venacava_moving)  # Load moving venacava label

        # Prepare result dictionary with loaded images
        result = {
            "moved_image": moving_image,  # Loaded moving image
            "moved_label": moving_label,  # Loaded moving label
            "moved_portalvein": moving_portalvein,  # Loaded moving portalvein label (if available)
            "moved_venacava": moving_venacava,  # Loaded moving venacava label (if available)
            "affine_matrix": None,  # Placeholder for affine matrix
            "svf": None,  # Placeholder for stationary velocity field
            "deformation": None  # Placeholder for deformation field
        }

        return result