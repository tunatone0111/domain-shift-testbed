from timm import create_model
from torchvision import transforms

from datamodule import num_classes_dict
from test_domain_shift import DomainShiftConfig, test_domain_shift


def main():
    for dataset_name, num_classes in num_classes_dict.items():
        model = create_model("resnet18.a1_in1k", num_classes=num_classes)
        cfg = DomainShiftConfig(
            log_dir=f"logs/res18-{dataset_name.split('/')[1]}",
            dataset_name=dataset_name,
            data_dir="/workspace/disk0/datasets",
            num_classes=num_classes,
            model=model,
            batch_size=64,
            num_workers=24,
            transform=transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            ),
        )
        test_domain_shift(cfg=cfg)


if __name__ == "__main__":
    main()
