from mnist_torch import mnistFCnet, get_data_loader
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

dump_folder = 'wrong_pred'

if __name__ == '__main__':
    trainloader = get_data_loader(1)
    model = mnistFCnet()
    model.load_state_dict(torch.load('saved-models/mnist_fc.pth'))
    model.eval()

    wrong_num = 0

    # clear the folder first
    if os.path.exists(dump_folder):
        for f in os.listdir(dump_folder):
            os.remove(dump_folder+'/'+f)
    else:
        os.mkdir(dump_folder)
    
    # find out which images are misclassified and save them with labels in a folder
    pbar = tqdm(trainloader)
    for images, labels in pbar:
        pbar.set_description(f'{wrong_num} wrong yet.')
        with torch.no_grad():
            logps = model(images)
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred = probab.index(max(probab))
        if pred != labels:
            wrong_num += 1
            # save the image with the label
            plt.imshow(images[0].numpy().squeeze(), cmap='Greys_r')
            plt.title(f'Predicted {pred} | Actual {labels.item()}')
            plt.savefig(dump_folder+f'/{labels.item()}-{pred}.png')
            plt.close()
        
    print(f'Complete. {wrong_num} images misclassified out of {len(trainloader)}.')