import datetime
import importlib
import yaml
import lightning as L
import gc
import lightning.pytorch.loggers as lightning_loggers
from torchvision import transforms
from src.data.after_image_dataset import AfterImageDataset
from src.data.before_after_image_with_box_dataset import BeforeAfterWithBoxImageDataset
import wandb
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from torch.utils.data import DataLoader
import torch
import os
import time
from torch.utils.data import Subset
from sklearn.model_selection import KFold


# Get the absolute path of the directory where the current script is located.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, "params.yaml")) as f:
    params: dict = yaml.safe_load(f)

with open(os.path.join(SCRIPT_DIR, "login_data.yaml")) as f:
    login_data: dict = yaml.safe_load(f)

def get_attr(module_attr: str):
    """
    Dynamically imports a class or function from a module using its string path.

    This allows for flexible component loading based on the configuration file.

    Args:
        module_attr (str): The full path to the attribute, in "module.path:ClassName" format.

    Returns:
        The class or function object.
    """

    assert (
        len(module_attr.split(":")) == 2
    ), "Invalid path format for attribute. It must be 'my_module:my_function'"

    module_name, attr_name = module_attr.split(":")

    module = importlib.import_module(module_name)

    return getattr(module, attr_name)

def construct_model():
    """
    Builds and returns the model instance based on settings in params.yaml.

    Returns:
        An instance of the specified model class.
    """
    model_path = params["model"]["path"]
    model = get_attr(model_path)

    return model(params)

def create_trainer(logger: lightning_loggers.Logger, callbacks: list):
    """
    Creates a PyTorch Lightning Trainer instance configured from params.yaml.

    Args:
        logger: The logger instance (e.g., WandbLogger) for experiment tracking.
        callbacks: A list of PyTorch Lightning callbacks (can be None).

    Returns:
        A configured L.Trainer instance.
    """
    return L.Trainer(logger=logger, callbacks=callbacks, **params["trainer"]["params"])

def get_transformations(set_name:str) -> list:
    """
    Builds a composition of image transformations for a given dataset split.

    Args:
        set_name (str): The name of the dataset split (e.g., 'train', 'val', 'test').

    Returns:
        A torchvision.transforms.Compose object containing the specified transformations.
    """
    transform_list = []
    if 'transformation' in params['data']:
        for transform_config in params['data']['transformation'][set_name]:
            print(f"Applying transformation: {transform_config}")
            transform_name = transform_config['name']

            # Collect all parameters for the transformation, excluding the 'name' key.
            transform_params = {k: v for k, v in transform_config.items() if k != 'name'}

            if hasattr(transforms, transform_name):
                transform_class = getattr(transforms, transform_name)
                transform_list.append(transform_class(**transform_params))
            else:
                print(f"Warning: Transformation '{transform_name}' not found in torchvision.transforms.")

    if not transform_list:
        raise ValueError(f"No transformations were configured for the '{set_name}' set.")

    return transforms.Compose(transform_list)

def get_dataloader():
    """
    Prepares and returns the DataLoaders for training, validation, and testing.

    This function handles the data splitting logic (train/val split or K-fold).

    Returns:
        A tuple of (train_loader, val_loader, test_loader).
    """

    if params['data']['kfolds']['active']:
        # Build the full dataset once
        full_dataset = BeforeAfterWithBoxImageDataset(params=params, transform=None)
        #full_dataset = AfterImageDataset(params, train=True, transform=None)
        if len(full_dataset) == 0:
            raise ValueError("Full dataset for KFold is empty. Check data path and loading logic.")

        if params['data']['additional_rating_label']['balanced']:
            full_dataset._level_out_labels()
            print("Balanced labels for KFold.")

        all_images = full_dataset.images
        all_labels = full_dataset.labels

        # Read KFold args (with sensible defaults)
        kf_cfg = params['data']['kfolds']
        n_splits = kf_cfg.get('n_splits', 5)
        shuffle = kf_cfg.get('shuffle', True)
        seed = params['data'].get('random_state', 42)

        # Try stratified CV when possible; fallback to plain KFold
        try:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed).split(all_images,
                                                                                                    all_labels)
        except Exception as e:
            print(f"StratifiedKFold unavailable ({e}). Falling back to KFold.")
            splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed).split(all_images)

        # Build transforms once
        train_transform = get_transformations('train')
        val_transform = get_transformations('val')
        test_transform = get_transformations('test')

        folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(splitter):
            print(f"Preparing KFold fold {fold_idx + 1}/{n_splits}...")

            train_images = [all_images[i] for i in train_idx]
            train_labels = [all_labels[i] for i in train_idx]
            val_images = [all_images[i] for i in val_idx]
            val_labels = [all_labels[i] for i in val_idx]

            train_dataset = BeforeAfterWithBoxImageDataset(params=params, images=train_images, labels=train_labels, transform=train_transform)
            val_dataset = BeforeAfterWithBoxImageDataset(params=params, images=val_images, labels=val_labels, transform=val_transform)
            #train_dataset = AfterImageDataset(params, images=train_images, labels=train_labels, train=True, transform=train_transform)
            #val_dataset = AfterImageDataset(params, images=val_images, labels=val_labels, train=True, transform=val_transform)


            train_loader = DataLoader(train_dataset, **params['dataloader']['train_loader'])
            val_loader = DataLoader(val_dataset, **params['dataloader']['val_loader'])
            folds.append((train_loader, val_loader))

        # Test set as before
        #test_set = BeforeAfterWithBoxImageDataset(params=params, transform=test_transform)
        #test_loader = DataLoader(test_set, **params['dataloader']['test_loader'])

        # Keep the same return arity
        return folds, None, None

    elif params['data']['train_val_split']['active']:

        split_params = params['data']['train_val_split']
        val_size = split_params['val_size']

        #full_dataset = AfterImageDataset(params, train=True, transform=None)
        full_dataset = BeforeAfterWithBoxImageDataset(params=params, transform=None)

        if len(full_dataset) == 0:
            raise ValueError("Full dataset for train/val split is empty. Check data path and loading logic.")

        if params['data']['additional_rating_label']['balanced']:
            full_dataset._level_out_labels()
            print("Balanced labels for training and validation sets.")

        all_images = full_dataset.images
        all_labels = full_dataset.labels
        if val_size > 0:
            try:
                train_images, val_images, train_labels, val_labels = train_test_split(
                    all_images,
                    all_labels,
                    test_size=val_size,
                    random_state=params['data']['random_state'],
                    shuffle=True,
                    stratify=all_labels
                )
            except ValueError as e:
                print(
                    f"Warning: Could not use stratification (possibly due to label format or class distribution). Splitting without stratification. Error: {e}")
                train_images, val_images, train_labels, val_labels = train_test_split(
                    all_images,
                    all_labels,
                    test_size=val_size,
                    random_state=params['data']['random_state'],
                    shuffle=True,
                    stratify=None  # Fallback to no stratification
                )
        else:
            print("No validation set configured. Using all data for training.")
            train_images, val_images, train_labels, val_labels = all_images, [], all_labels, []


        train_transform = get_transformations('train')
        val_transform = get_transformations('val')
        test_transform = get_transformations('test')

        print("Creating train and validation dataset instances with transformations...")
        '''train_dataset = AfterImageDataset(params, images=train_images, labels=train_labels, train=True, transform=train_transform)
        val_dataset = AfterImageDataset(params, images=val_images, labels=val_labels, train=True, transform=val_transform)

        test_set = AfterImageDataset(params, train=False, transform=test_transform)'''

        train_dataset = BeforeAfterWithBoxImageDataset(params=params, images=train_images, labels=train_labels, transform=train_transform)
        if val_size > 0:
            val_dataset = BeforeAfterWithBoxImageDataset(params=params, images=val_images, labels=val_labels, transform=val_transform)

        #test_set = BeforeAfterWithBoxImageDataset(params=params, transform=test_transform)


        train_loader = DataLoader(train_dataset, **params['dataloader']['train_loader'])
        if val_size > 0:
            val_loader = DataLoader(val_dataset, **params['dataloader']['val_loader'])
        else:
            val_loader = None
        #test_loader = DataLoader(test_set, **params['dataloader']['test_loader'])
        test_loader = None

        return train_loader, val_loader, test_loader
    else:
        raise ValueError("No valid data splitting method is configured in params.yaml. "
                         "Please check the 'kfolds' and 'train_val_split' sections.")

def get_logger(**overrides):
    """
    Initializes and returns the logger for experiment tracking.

    Returns:
        A configured Lightning logger instance (e.g., WandbLogger).
    """
    wandb.login(key=login_data["wandb"]["api_key"])
    logger_params = {**params["logger"]["params"], **overrides}
    return getattr(lightning_loggers, params["logger"]["lightning_logger_name"])(
        **logger_params
    )


if __name__ == "__main__":

    train_loader, val_loader, test_loader = get_dataloader()
    gc.collect()
    if params['data']['kfolds']['active']:
        folds = train_loader  # list of (train_loader, val_loader)
        n_splits = len(folds)

        # A stable group keeps all fold runs together in W&B UI
        base_group = params["logger"]["params"].get("group", f"cv-{n_splits}-folds")

        for fold_idx, (tr_loader, va_loader) in enumerate(folds, start=1):
            # Fresh seed & model per fold
            L.seed_everything(params['data'].get('random_state', 42), workers=True)
            model = construct_model()

            # Clean W&B run naming
            run_name = f"{params['logger']['params'].get('name', 'run')}-fold{fold_idx}/{n_splits}"
            logger = get_logger(name=run_name, group=base_group)

            # Trainer per fold
            trainer = create_trainer(logger=logger, callbacks=None)

            # Optional: log the fold in the run config for clarity
            try:
                logger.experiment.config.update({"fold": fold_idx, "n_splits": n_splits}, allow_val_change=True)
            except Exception:
                pass

            # Train
            if params['trainer']['params'].get('overfit_batches', 0) > 0:
                trainer.fit(model, tr_loader)
            else:
                trainer.fit(model, tr_loader, va_loader)

            # (Optional) save a checkpoint per fold
            if params['model']['save_model']:
                print(f"Saving trained model for fold {fold_idx}...")
                model_path = os.path.join(os.getcwd(),
                                          f"{time.datetime.now().strftime('%Y%m%d_%H%M%S')}_fold{fold_idx}_model.pth")
                trainer.save_checkpoint(model_path)

            # End the W&B run cleanly so each fold is its own run
            wandb.finish()

            torch.cuda.empty_cache()

            del trainer
            del model
            del logger

            del tr_loader, va_loader
            gc.collect()

    else:

        model = construct_model()

        logger = get_logger()

        trainer = create_trainer(logger = logger, callbacks = None)

        if params['trainer']['params'].get('overfit_batches', 0) > 0:
            trainer.fit(model, train_loader)
        else:
            trainer.fit(model, train_loader, val_loader)

        if params['model']['save_model']:
            print("Saving the trained model...")
            model_path = os.path.join(os.getcwd(), datetime.now().strftime("%Y%m%d_%H%M%S") + "_model.pth")
            trainer.save_checkpoint(model_path)