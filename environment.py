import torch

def to_patches(x, region_size):  # x.shape = (N, C, H, W)
    k = x.shape[1]
    x = x.unfold(2, region_size, region_size).unfold(3, region_size, region_size)
    x = x.permute(0, 2, 3, 1, 4, 5)
    x = x.reshape(-1, k, region_size, region_size)
    return x

class Environment:
    def __init__(self, seg_model, nregions, device):
        self.seg_model = torch.load(seg_model, map_location=device)
        self.seg_model.eval()
        self.seg_model.backbone.register_forward_hook(self._latent_space_hook)
        self.device = device
        self.nregions = nregions

    def _latent_space_hook(self, module, args, output):
        self.latent = output['out'].mean([2, 3])  # (1, 2048, 32, 32) -> (1, 2048)

    def _calc_reward(self, mask):
        return (mask == 1).float().mean()
        # should we divide by the number of iterations?

    def reset(self, image, mask):
        region_size = image.shape[1] // self.nregions
        self.image_regions = to_patches(image[None], region_size)
        self.mask = mask.long()[None, None]
        self.state = torch.zeros((1, 2048, self.nregions, self.nregions), device=self.device)
        self.region_selected = torch.zeros(self.nregions**2, dtype=bool)
        self.iteration = 0
        self.output = torch.zeros((1, 1, image.shape[1], image.shape[2]), dtype=torch.int32)

    def possible_actions(self):
        return self.nregions**2

    def was_region_selected(self, action):
        return self.region_selected[action]

    def get_state(self):
        return self.state
    
    def step(self, action):
        assert 0 <= action < self.possible_actions()
        self.iteration += 1
        if self.was_region_selected(action):
            return 0, False
        preds = self.seg_model(self.image_regions[[action]])['out']  # (1, 1, 256, 256)
        preds = torch.sigmoid(preds) >= 0.5
        x = action % self.nregions
        y = action // self.nregions
        self.state[..., y, x] = self.latent[0]
        reward = self._calc_reward(preds)
        isover = torch.all(self.region_selected)
        self.output[:, :, y*preds.shape[2]:(y+1)*preds.shape[2],
            x*preds.shape[3]:(x+1)*preds.shape[3]] = preds
        score = (self.output == self.mask).float().mean()
        return reward, isover, score

if __name__ == '__main__':  # DEBUG
    from skimage.io import imread 
    from skimage.transform import resize
    image = resize(imread('x-image.bmp'), (2048, 2048))
    mask = resize(imread('x-mask.bmp'), (2048, 2048))
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    mask = torch.tensor(mask)
    env = Environment('model-seg.pth', 8, 'cpu')
    env.reset(image, mask)
    #for i in range(8):
    #    import matplotlib.pyplot as plt
    #    plt.imshow(env.image_regions[i].permute(1, 2, 0))
    #    plt.savefig(f'debug-{i}.jpg')
    print('possible actions:', env.possible_actions())
    print('was region selected:', env.was_region_selected(5))
    print('get state:', env.get_state().shape)
    for i in range(64):
        print('step:', i, env.step(i))