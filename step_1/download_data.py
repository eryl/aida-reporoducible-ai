from torchvision.datasets import OxfordIIITPet

dataset = OxfordIIITPet('data/oxfordiii', download=True, split='trainval')
