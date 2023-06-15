import requests

resp = requests.post(
    "http://127.0.0.1:5000",
    files={
        "file": open(
            r"D:\bangkit\scantion-ml-model-main\Dataset\test\malignant\ISIC_0033779.jpg",
            "r",
        )
    },
)

print(resp.json())
