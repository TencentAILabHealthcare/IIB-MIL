def build_dataset(input_data_file, image_set, args):
    dataset_name = args.DATASET.DATASET_NAME

    if dataset_name in ("patch_confidence"):
        from .wsi_feat_dataset import (
            build as build_patch_dataset_confidence,
        )

        return build_patch_dataset_confidence(input_data_file, image_set, args)

    raise ValueError(f"dataset {dataset_name} not supported")
